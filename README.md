# Mini-SpikingBrain

A small PyTorch reimplementation of the core ideas in
[SpikingBrain Technical Report: Spiking Brain-inspired Large Models](https://arxiv.org/abs/2509.05276),
scaled down to run on a laptop (Apple Silicon / MPS, a single CUDA GPU, or CPU).

The goal is to **reproduce the mechanisms** of the paper and measure their trade-offs,
not to match its absolute numbers:

| Paper idea | Where in this repo |
|---|---|
| Hybrid efficient attention: full softmax (FA), sliding window (SWA), linear (LA) — Eq. (1)–(4) | `attention.py` |
| Attention map correspondence: reuse pretrained Q/K/V/O weights to convert FA → hybrid (Sec. 3.1) | `train.py --stage convert` |
| MoE upcycling with output rescaling α = (1 / (S + k/N))^(1/3) (Sec. 3.2, Eq. 15–18) | `moe.py`, `train.py --stage upcycle` |
| Adaptive threshold spiking: V_th(x) = mean(\|x\|)/k, s_INT = round(x / V_th) (Step 1) | `layers.py` (`AdaptiveSpike`, `SpikingLinear`) |
| Spike coding: ternary and bidirectional bitwise, event-driven matmul (Step 2, Eq. 22–23) | `spike_coding.py` |
| Sparsity / spike count / energy estimates | `eval.py` |

---

## Project structure

```
mini_spikingbrain/
├── data.py          # FineWeb-Edu: stream, tokenize (GPT-2 BPE), write memory-mapped .bin files
├── layers.py        # RMSNorm, SwiGLU, AdaptiveSpike (STE), SpikingLinear
├── attention.py     # Attention (fa / swa / la), HybridAttention, la_recurrent
├── moe.py           # SparseMoE (sigmoid router, shared expert, aux loss), upcycle()
├── model.py         # GPTConfig, Block, GPT (attention type and FFN/MoE chosen per layer)
├── spike_coding.py  # ternary_encode, bitwise_encode, event_driven_linear (+ self-test)
├── train.py         # stages: baseline | convert | upcycle | spike | finetune
├── eval.py          # val loss, spike stats, coding cost, equivalence check, expert load, samples
└── plot_logs.py     # plot loss curves from logs/*.json
```

Generated (git-ignored): `data/` (tokenized dataset), `ckpt/` (checkpoints).
Kept in git: `logs/` (loss curves), `results/` (eval output).

---

## Setup

```bash
pip install torch numpy datasets transformers tqdm matplotlib
```

### Prepare data (once, ~2 minutes, ~200 MB on disk)

```bash
python data.py --out data/fineweb_edu --tokenizer gpt2 --n_tokens 100_000_000
python data.py --out data/fineweb_edu --check      # verify an existing dataset
```

Streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`), tokenizes with GPT-2, and writes
`train.bin`, `val.bin`, `meta.json`. Documents are split train/val at document level
(every 200th document → val) and separated by EOS.

> If you later convert a pretrained model, prepare the data with **that model's tokenizer**
> (e.g. `--tokenizer HuggingFaceTB/SmolLM2-135M`).

### Sanity checks

```bash
python spike_coding.py        # ternary / bitwise must match integer matmul (max_err ≈ 0)
python train.py --stage baseline --run_name smoke --steps 20 --eval_every 10 --eval_iters 5
```

The first validation loss should be ≈ ln(50257) ≈ 10.8.

---

## Training pipeline

```
baseline (FA) ──convert──► hybrid LA+SWA ──upcycle──► + MoE ──spike──► spiking model
```

Every stage writes `ckpt/<run_name>.pt` and `logs/<run_name>.json` (including the val loss at step 0,
which shows how close a converted / upcycled model starts to its parent).

Small config used in the experiments below (shared by all commands):

```bash
CFG="--batch_size 16 --block_size 256 --window 64 --d_model 256 --n_layer 4 --n_head 4 --ffn_hidden 688"
```

| Stage | Command | Purpose |
|---|---|---|
| baseline | `python train.py --stage baseline --run_name baseline_s --steps 2000 --warmup 200 $CFG` | FA Transformer from scratch (stands in for the "open-source" base model) |
| finetune | `python train.py --stage finetune --init ckpt/baseline_s.pt --run_name baseline_s_ft --steps 500 --warmup 50` | **control run**: same extra steps, no architecture change |
| spike | `python train.py --stage spike --init ckpt/baseline_s.pt --spike_k 1.0 --run_name spike_k1 --steps 500 --warmup 50` | wrap all Linear layers (except `head`, `router`) with `SpikingLinear`, then fine-tune |
| convert | `python train.py --stage convert --init ckpt/baseline_s.pt --run_name convert --steps 1000` | hybrid attention initialized from the baseline (`--hybrid_style intra` or `inter`) |
| convert (control) | `python train.py --stage convert --run_name convert_scratch --steps 1000 $CFG` | same hybrid trained from scratch |
| upcycle | `python train.py --stage upcycle --init ckpt/convert.pt --run_name upcycle --steps 1000` | dense FFN → MoE (8 experts, top-1, 1 shared); add `--no_alpha` for the ablation |

Useful flags: `--max_train_tokens N` (use only the first N training tokens), `--lr`, `--seed`.
Stages after `baseline` default to a lower learning rate (3e-4), following Sec. 3.1.

> **Compare like with like.** Any stage that trains extra steps must be compared against a
> control trained for the same number of steps (`--stage finetune`, `convert_scratch`, …);
> otherwise the gain from extra training is mixed into the effect being measured.

---

## Evaluation

```bash
# compare checkpoints on the SAME validation batches (fixed numpy + torch seed)
python eval.py --ckpts ckpt/baseline_s.pt ckpt/baseline_s_ft.pt ckpt/spike_k1.pt --eval_iters 100

# spike a trained model WITHOUT fine-tuning, sweeping k
python eval.py --ckpts ckpt/baseline_s.pt --spike_sweep 0.5 1 2 4 --check_equiv

# generate text
python eval.py --ckpts ckpt/baseline_s.pt --sample 200

# loss curves
python plot_logs.py --train
```

`eval.py` reports, per checkpoint:

| Field | Meaning |
|---|---|
| `val` | validation loss |
| `zero` | fraction of integer spike counts s_INT equal to 0 |
| `spk/act` | mean \|s_INT\| per activation (≈ k by construction of the adaptive threshold) |
| `in±7` | fraction of counts in [−7, 7] (paper: ~94%) |
| `max\|s\|` | largest count (outlier "burst") |
| `tern: T̄`, `bit: T̄` | average time steps per layer for ternary vs bitwise coding |
| `E` | estimated energy vs dense FP16 matmul (additions × 0.03 pJ vs MACs × 1.5 pJ) |
| `equiv err` | relative error between event-driven Eq. (22)/(23) and integer matmul |

The energy numbers are a **theoretical estimate for event-driven hardware** with assumed constants
(override with `--e_mac`, `--e_add`). Sparsity does **not** speed up GPU/MPS execution.

---

## Results so far

Small config (4 layers, d=256, 16.1M params of which ~12.9M are the tied embedding),
GPT-2 tokenizer, FineWeb-Edu. All rows evaluated on the same 100 validation batches.

### Spiking after training (no fine-tuning), `baseline_s` val loss = 5.803

| k | val loss | Δ vs baseline | zeros | spikes / act | in ±7 | energy (ternary / bitwise) |
|---|---|---|---|---|---|---|
| 1 | 5.912 | +0.109 | 37.2% | 0.96 | 99.7% | 1.93% / 1.39% |
| 2 | 5.832 | +0.029 | 21.0% | 1.98 | 98.2% | 3.97% / 2.06% |

An earlier sweep (different val batches) also gave k = 0.5 → 6.16 and k = 4 → 5.82.

### Spiking with fine-tuning (500 extra steps, batch 32)

| Model | val loss | Δ vs control |
|---|---|---|
| `baseline_s_ft` (control, no spiking) | 5.668 | — |
| `spike_k1` | 5.750 | +0.081 |

### Findings

1. **Spike coding is exact.** Ternary and bitwise event-driven computation match integer matmul
   up to float32 rounding (relative error ~1e-6).
2. **The adaptive threshold controls firing precisely.** Mean spikes per activation ≈ k, and it
   stays unchanged after fine-tuning.
3. **k is the main trade-off knob.** k = 2 loses only +0.029 without fine-tuning, less than k = 1
   after fine-tuning (+0.081), at ~2% of dense energy (estimated).
4. **Fine-tuning recovers only part of the loss** at k = 1 (+0.109 → +0.081, ~26%).
5. **Bitwise coding matters for outliers.** A few activations reach 90–440 spikes; ternary then
   needs ~20–85 steps per layer, bitwise ~4–6.
6. Zero-fraction here (37% at k = 1) is measured on integer counts, not on expanded spike trains
   as in the paper (69%), so the two numbers are not directly comparable.

---
## Limitations

- Context is short (256 tokens), so the O(n) speed/memory advantage of LA/SWA is not visible;
  only quality is compared.
- LA is computed in its parallel O(n²) form for training; `la_recurrent` shows the O(1)-state form.
- The baseline is trained from scratch on ~8M tokens, so it holds far less knowledge than the
  pretrained models converted in the paper.

## Reference

Pan et al., *SpikingBrain Technical Report: Spiking Brain-inspired Large Models*, arXiv:2509.05276, 2025.
Code from the authors: https://github.com/BICLab/SpikingBrain-7B
