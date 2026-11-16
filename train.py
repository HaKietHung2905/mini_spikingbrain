"""
train.py — Train Mini-SpikingBrain in 4 stages

  baseline : FA Transformer, trained from scratch               (the "open-source" base model)
  convert  : hybrid LA+SWA, Q/K/V/O + FFN initialized from base (Section 3.1)
  upcycle  : replace FFN with Sparse MoE, scaled by alpha       (Section 3.2)
  spike    : wrap nn.Linear with SpikingLinear, then fine-tune  (Adaptive Threshold Spiking)

Examples:
  python train.py --stage baseline --steps 5000
  python train.py --stage convert  --init ckpt/baseline.pt --steps 1500
  python train.py --stage convert  --steps 1500 --run_name convert_scratch      # control run (no --init)
  python train.py --stage upcycle  --init ckpt/convert.pt --steps 1500
  python train.py --stage upcycle  --init ckpt/convert.pt --steps 1500 --no_alpha --run_name upcycle_noalpha
  python train.py --stage spike    --init ckpt/upcycle.pt --spike_k 1.0 --steps 500 --run_name spike_k1

Assumptions about the other files:
  data.py  : load_data(path) -> (train_ids, val_ids, meta) ; meta["vocab_size"], meta["decode"]
             get_batch(data, batch_size, block_size, device) -> (x, y)
  model.py : GPTConfig (dataclass) with fields:
                 vocab_size, block_size, d_model, n_layer, n_head, ffn_hidden, window,
                 attn_modes (list[str]: "fa" | "swa" | "la" | "hybrid_swa" | "hybrid_fa"),
                 moe_layers (list[int]), n_experts, top_k, n_shared, spike_k (float | None)
             GPT(cfg): model.blocks[i].attn, model.blocks[i].ffn
                       forward(idx, targets=None) -> (logits, loss)
                       if layer i is in cfg.moe_layers, GPT builds a SparseMoE instead of SwiGLU
  moe.py   : upcycle(dense_ffn, d, n_exp, top_k, n_shared, use_alpha=True)
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace

import torch
import torch.nn as nn

from data import load_data, get_batch
from model import GPT, GPTConfig
from layers import SpikingLinear
from moe import upcycle

SKIP_SPIKE = ("head", "router")   # do not spike the vocab output layer or the MoE router


# ============================================================
# Model / checkpoint helpers (also imported by eval.py)
# ============================================================
def spikify(model: nn.Module, k: float) -> nn.Module:
    """Wrap every nn.Linear (except SKIP_SPIKE) with SpikingLinear: spikes the INPUT of the projection."""
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, SpikingLinear):          # already wrapped, skip
            continue
        for cname, child in module.named_children():
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and not any(s in full for s in SKIP_SPIKE):
                targets.append((module, cname, child))
    for parent, cname, child in targets:
        setattr(parent, cname, SpikingLinear(child, k))
    print(f"[spikify] wrapped {len(targets)} Linear layers, k={k}")
    return model


def build_model(cfg: GPTConfig) -> nn.Module:
    """Rebuild the exact architecture from cfg (incl. MoE and spiking) so load_state_dict keys match."""
    model = GPT(cfg)
    if cfg.spike_k is not None:
        spikify(model, cfg.spike_k)
    return model


def save_ckpt(path, model, cfg, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"cfg": asdict(cfg), "model": model.state_dict(), **(extra or {})}, path)
    print(f"[save] {path}")


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = GPTConfig(**ck["cfg"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device), cfg


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device, iters=50):
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


# ============================================================
# Per-stage configuration
# ============================================================
def base_cfg(args, vocab_size) -> GPTConfig:
    return GPTConfig(
        vocab_size=vocab_size, block_size=args.block_size,
        d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
        ffn_hidden=args.ffn_hidden, window=args.window,
        attn_modes=["fa"] * args.n_layer, moe_layers=[],
        n_experts=args.n_experts, top_k=args.top_k, n_shared=args.n_shared,
        spike_k=None,
    )


def hybrid_modes(n_layer, style):
    if style == "intra":   # 76B-style: LA+SWA in parallel in every layer, last layer LA+FA
        return ["hybrid_swa"] * (n_layer - 1) + ["hybrid_fa"]
    # 7B-style (inter-layer): alternate LA / SWA, no FA
    return ["la" if i % 2 == 0 else "swa" for i in range(n_layer)]


def prepare_model(args, vocab_size):
    if args.stage == "baseline":
        cfg = base_cfg(args, vocab_size)
        return GPT(cfg), cfg

    if args.stage == "convert":
        ck = None
        cfg = base_cfg(args, vocab_size)
        if args.init:                                   # take model size from the base checkpoint
            ck = torch.load(args.init, map_location="cpu", weights_only=True)
            cfg = GPTConfig(**ck["cfg"])
        cfg = replace(cfg, attn_modes=hybrid_modes(cfg.n_layer, args.hybrid_style),
                      window=args.window)
        model = GPT(cfg)
        if ck is not None:
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            print("[convert] NEW params (randomly initialized):", missing)
            print("[convert] ignored params:", unexpected)
        else:
            print("[convert] no --init -> hybrid trained FROM SCRATCH (control run)")
        return model, cfg

    if args.stage == "upcycle":
        assert args.init, "--stage upcycle requires --init (dense checkpoint)"
        model, cfg = load_ckpt(args.init)
        # default: keep layer 0 dense (the paper keeps early layers dense)
        moe_layers = args.moe_layers if args.moe_layers else list(range(1, cfg.n_layer))
        for i in moe_layers:
            model.blocks[i].ffn = upcycle(
                model.blocks[i].ffn, cfg.d_model,
                n_exp=args.n_experts, top_k=args.top_k, n_shared=args.n_shared,
                use_alpha=not args.no_alpha,
            )
        cfg = replace(cfg, moe_layers=moe_layers, n_experts=args.n_experts,
                      top_k=args.top_k, n_shared=args.n_shared)
        print(f"[upcycle] MoE at layers {moe_layers}, alpha={'on' if not args.no_alpha else 'off'}")
        return model, cfg

    if args.stage == "finetune":
        # control run: continue training a checkpoint with NO architecture change
        assert args.init, "--stage finetune requires --init"
        return load_ckpt(args.init)

    if args.stage == "spike":
        assert args.init, "--stage spike requires --init"
        model, cfg = load_ckpt(args.init)
        spikify(model, args.spike_k)
        cfg = replace(cfg, spike_k=args.spike_k)
        return model, cfg

    raise ValueError(args.stage)


# ============================================================
# Training loop
# ============================================================
def lr_at(step, args):
    if step < args.warmup:                                   # linear warm-up
        return args.lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["baseline", "convert", "upcycle", "spike", "finetune"])
    ap.add_argument("--init", default=None, help="checkpoint to initialize from")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--data", default="data/fineweb_edu")
    ap.add_argument("--max_train_tokens", type=int, default=None,
                    help="use only the first N train tokens (quick experiments)")
    ap.add_argument("--out_dir", default="ckpt")
    ap.add_argument("--log_dir", default="logs")
    # model
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--ffn_hidden", type=int, default=1024)  # ~ 8/3 * d, SwiGLU-style
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--hybrid_style", default="intra", choices=["intra", "inter"])
    # MoE
    ap.add_argument("--moe_layers", type=int, nargs="*", default=None)
    ap.add_argument("--n_experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--n_shared", type=int, default=1)
    ap.add_argument("--no_alpha", action="store_true", help="ablation: upcycle without alpha scaling")
    # spike
    ap.add_argument("--spike_k", type=float, default=1.0)
    # optim
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 for baseline, 3e-4 for other stages")
    ap.add_argument("--min_lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # conversion uses a lower LR than pre-training (Section 3.1, point 2)
    args.lr = args.lr or (1e-3 if args.stage == "baseline" else 3e-4)
    args.min_lr = args.min_lr if args.min_lr is not None else args.lr * 0.1
    run = args.run_name or args.stage

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    train_data, val_data, meta = load_data(args.data, args.max_train_tokens)
    model, cfg = prepare_model(args, meta["vocab_size"])
    model.to(device)
    bs = cfg.block_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{run}] device={device} params={n_params/1e6:.2f}M attn={cfg.attn_modes}")

    # full-parameter training, backbone not frozen (Section 3.1, point 4)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)

    log = {"run": run, "stage": args.stage, "args": vars(args),
           "step": [], "train_loss": [], "val_loss": []}

    # loss at step 0: key for the convert / upcycle experiments (is the new model ~ the base model?)
    v0 = estimate_loss(model, val_data, args.batch_size, bs, device, args.eval_iters)
    print(f"[{run}] step 0 | val {v0:.4f}")
    log["step"].append(0); log["train_loss"].append(None); log["val_loss"].append(v0)

    model.train()
    t0, running = time.time(), 0.0
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, args)

        x, y = get_batch(train_data, args.batch_size, bs, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running += loss.item()

        if step % args.eval_every == 0 or step == args.steps:
            n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
            tr = running / n; running = 0.0
            va = estimate_loss(model, val_data, args.batch_size, bs, device, args.eval_iters)
            log["step"].append(step); log["train_loss"].append(tr); log["val_loss"].append(va)
            print(f"[{run}] step {step:5d} | train {tr:.4f} | val {va:.4f} | "
                  f"lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")

    save_ckpt(os.path.join(args.out_dir, f"{run}.pt"), model, cfg)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, f"{run}.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()