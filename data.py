"""
data.py — FineWeb-Edu data pipeline (token-level, memory-mapped)

Two parts:

1) PREPARE (run once): stream FineWeb-Edu, tokenize, write uint16/uint32 .bin files
     python data.py --out data/fineweb_edu --tokenizer gpt2 --n_tokens 100_000_000
   Output directory:
     train.bin, val.bin   flat token streams, documents separated by EOS
     meta.json            tokenizer name, vocab_size, dtype, token counts

2) LOAD (used by train.py / eval.py), same interface as before:
     load_data(path) -> (train_ids, val_ids, meta)
         train_ids, val_ids : np.memmap of token ids (never fully loaded into RAM)
         meta : {"vocab_size", "tokenizer_name", "encode", "decode", "eos_id"}
     get_batch(data, batch_size, block_size, device) -> (x, y)

Notes:
  - Use the SAME tokenizer as the pretrained model if you later convert one
    (e.g. --tokenizer HuggingFaceTB/SmolLM2-135M); token ids must match its embedding.
  - Dependencies: numpy, datasets, transformers, tqdm
"""
import argparse
import json
import os

import numpy as np
import torch

DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"


# ============================================================
# Tokenizer helpers
# ============================================================
def _load_tokenizer(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _dtype_for(vocab_size):
    # uint16 holds ids < 65536 (GPT-2, SmolLM2); larger vocabs (e.g. Qwen) need uint32
    return np.uint16 if vocab_size < 2 ** 16 else np.uint32


# ============================================================
# 1) PREPARE
# ============================================================
def prepare(out_dir, tokenizer_name, n_tokens, val_every, batch_docs, flush_tokens):
    from datasets import load_dataset
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)
    tok = _load_tokenizer(tokenizer_name)
    # we only tokenize to disk; silence the "sequence longer than model max length" warning
    tok.model_max_length = int(1e12)
    vocab_size = len(tok)                         # includes added special tokens
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer has no EOS token; set one before preparing"
    dtype = _dtype_for(vocab_size)

    ds = load_dataset(DATASET, name=SUBSET, split="train", streaming=True)

    paths = {s: os.path.join(out_dir, f"{s}.bin") for s in ("train", "val")}
    files = {s: open(p, "wb") for s, p in paths.items()}
    buffers = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    buffered = {"train": 0, "val": 0}

    def flush(split):
        if buffers[split]:
            np.concatenate(buffers[split]).astype(dtype).tofile(files[split])
            buffers[split].clear()
            buffered[split] = 0

    pbar = tqdm(total=n_tokens, unit="tok", desc="train tokens")
    doc_idx = 0
    batch = []

    def process(texts, start_idx):
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for j, ids in enumerate(enc):
            # deterministic document-level split: no document is shared by train and val
            split = "val" if (start_idx + j) % val_every == 0 else "train"
            arr = np.asarray(ids + [eos], dtype=np.int64)
            buffers[split].append(arr)
            buffered[split] += len(arr)
            counts[split] += len(arr)
            if split == "train":
                pbar.update(len(arr))
            if buffered[split] >= flush_tokens:
                flush(split)

    for ex in ds:
        batch.append(ex["text"])
        if len(batch) == batch_docs:              # batched tokenization is much faster
            process(batch, doc_idx)
            doc_idx += len(batch)
            batch = []
            if counts["train"] >= n_tokens:
                break
    if batch and counts["train"] < n_tokens:
        process(batch, doc_idx)

    for s in files:
        flush(s)
        files[s].close()
    pbar.close()

    meta = {"dataset": f"{DATASET}/{SUBSET}", "tokenizer_name": tokenizer_name,
            "vocab_size": vocab_size, "eos_id": eos, "dtype": np.dtype(dtype).name,
            "train_tokens": counts["train"], "val_tokens": counts["val"]}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] wrote {counts['train']:,} train / {counts['val']:,} val tokens to {out_dir}")


# ============================================================
# 2) LOAD
# ============================================================
def load_data(path="data/fineweb_edu", max_train_tokens=None):
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    dtype = np.dtype(meta["dtype"])
    train_ids = np.memmap(os.path.join(path, "train.bin"), dtype=dtype, mode="r")
    if max_train_tokens:                                  # use only the first N tokens (still a memmap view)
        train_ids = train_ids[:max_train_tokens]
    val_ids = np.memmap(os.path.join(path, "val.bin"), dtype=dtype, mode="r")

    tok = _load_tokenizer(meta["tokenizer_name"])
    meta["encode"] = lambda s: tok.encode(s, add_special_tokens=False)
    meta["decode"] = lambda ids: tok.decode(ids)

    print(f"[data] {meta['dataset']} | tokenizer {meta['tokenizer_name']} | "
          f"vocab {meta['vocab_size']} | train {len(train_ids):,} | val {len(val_ids):,} tokens")
    return train_ids, val_ids, meta


def get_batch(data, batch_size, block_size, device="cpu"):
    """Sample random windows from the token stream; y is x shifted left by one token."""
    assert len(data) > block_size + 1, "dataset is shorter than block_size"
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    # copy each slice out of the memmap and cast to int64 (embedding expects LongTensor)
    x = torch.from_numpy(np.stack([data[i:i + block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size].astype(np.int64) for i in ix]))
    if str(device).startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare FineWeb-Edu token files")
    ap.add_argument("--out", default="data/fineweb_edu")
    ap.add_argument("--tokenizer", default="gpt2",
                    help="HF tokenizer name; use the pretrained model's tokenizer if you plan to convert it")
    ap.add_argument("--n_tokens", type=int, default=100_000_000, help="target number of TRAIN tokens")
    ap.add_argument("--val_every", type=int, default=200, help="every N-th document goes to val (~0.5%%)")
    ap.add_argument("--batch_docs", type=int, default=1000, help="documents per tokenizer call")
    ap.add_argument("--flush_tokens", type=int, default=10_000_000, help="flush buffer to disk every N tokens")
    ap.add_argument("--check", action="store_true", help="only load an existing output and print a sample")
    args = ap.parse_args()

    if not args.check:
        prepare(args.out, args.tokenizer, args.n_tokens, args.val_every,
                args.batch_docs, args.flush_tokens)

    # sanity check
    train_ids, val_ids, meta = load_data(args.out)
    x, y = get_batch(train_ids, batch_size=2, block_size=64)
    assert torch.equal(x[:, 1:], y[:, :-1]), "targets must be inputs shifted by one"
    print("sample:", repr(meta["decode"](x[0].tolist())))

    # Breaking out of a streaming dataset can leave background download threads alive,
    # which keeps the interpreter from exiting. All files are closed by now, so exit hard.
    import sys
    sys.stdout.flush()
    os._exit(0)