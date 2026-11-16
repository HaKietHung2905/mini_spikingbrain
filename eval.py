"""
eval.py — Evaluate Mini-SpikingBrain

For each checkpoint:
  1. validation loss
  2. spike statistics (if the model has SpikingLinear): % zeros, avg spikes / activation, % in [-7, 7]
  3. ternary vs bitwise coding cost: number of spikes (additions), time steps, estimated energy
  4. equivalence check for Eq. (22)/(23): event-driven result == integer matmul
  5. token distribution across experts (if the model has MoE)

Examples:
  python eval.py --ckpts ckpt/baseline.pt ckpt/convert.pt ckpt/convert_scratch.pt
  python eval.py --ckpts ckpt/spike_k1.pt --check_equiv --sample 300
  # spike AFTER training, no fine-tuning, sweep k (quick version of experiment 3):
  python eval.py --ckpts ckpt/upcycle.pt --spike_sweep 0.5 1 2 4

Energy is only a THEORETICAL estimate for event-driven hardware:
  E_dense = #MACs x E_MAC_FP16 ; E_spike = #additions x E_ADD_INT8
  Default constants follow Horowitz-style (45nm) numbers — state these assumptions when reporting.
"""
import argparse
import json
import os

import numpy as np
import torch

from data import load_data, get_batch
from layers import AdaptiveSpike, SpikingLinear
from moe import SparseMoE
from spike_coding import ternary_encode, bitwise_encode, event_driven_linear
from train import load_ckpt, spikify, estimate_loss


def seed_all(seed):
    # get_batch samples with numpy, so numpy must be seeded too (torch alone is not enough)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Probe: run one fixed batch so every AdaptiveSpike stores s_int and vth
# ============================================================
@torch.no_grad()
def probe(model, data, batch_size, block_size, device, seed=0):
    # clear values left over from estimate_loss (e.g. experts not hit in this batch)
    for m in model.modules():
        if isinstance(m, AdaptiveSpike):
            m.__dict__.pop("s_int", None); m.__dict__.pop("vth", None)
    seed_all(seed)
    x, y = get_batch(data, batch_size, block_size, device)
    model.eval()
    model(x, y)


def spiking_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, SpikingLinear)]


# ============================================================
# 1. Spike statistics (Adaptive Threshold, Step 1)
# ============================================================
def spike_stats(model):
    total = zeros = abs_sum = in_pm7 = 0
    max_abs = 0
    for _, m in model.named_modules():
        if isinstance(m, AdaptiveSpike) and hasattr(m, "s_int"):
            s = m.s_int.round()
            total += s.numel()
            zeros += (s == 0).sum().item()
            abs_sum += s.abs().sum().item()
            in_pm7 += (s.abs() <= 7).sum().item()
            max_abs = max(max_abs, int(s.abs().max()))
    if total == 0:
        return None
    return {
        "sparsity": zeros / total,            # paper: ~69%
        "avg_spikes": abs_sum / total,        # paper: ~1.13
        "in_pm7": in_pm7 / total,             # paper: ~94%
        "max_abs_count": max_abs,             # largest outlier (burst)
    }


# ============================================================
# 2. Coding cost: ternary vs bitwise (Step 2)
# ============================================================
def popcount(mag: torch.Tensor) -> torch.Tensor:
    """Number of 1-bits per element (mag: long, non-negative)."""
    out = torch.zeros_like(mag)
    for b in range(max(int(mag.max()).bit_length(), 1)):
        out += (mag >> b) & 1
    return out


def coding_cost(model, e_mac, e_add):
    macs = tern_adds = bit_adds = 0
    tern_steps, bit_steps = [], []
    for _, m in spiking_layers(model):
        if not hasattr(m.spike, "s_int"):
            continue
        mag = m.spike.s_int.round().abs().long()
        out_f = m.lin.out_features
        macs += mag.numel() * out_f                    # dense matmul: every input x every output
        tern_adds += mag.sum().item() * out_f          # each ±1 spike triggers out_f additions
        bit_adds += popcount(mag).sum().item() * out_f # bitwise: only 1-bits are spikes
        mx = int(mag.max())
        tern_steps.append(mx)                          # ternary needs |s|_max steps
        bit_steps.append(max(mx.bit_length(), 1))      # bitwise needs log2 steps
    if macs == 0:
        return None
    e_dense = macs * e_mac
    return {
        "dense_MACs": macs,
        "ternary": {"adds": tern_adds, "avg_steps": sum(tern_steps) / len(tern_steps),
                    "max_steps": max(tern_steps), "energy_ratio": tern_adds * e_add / e_dense},
        "bitwise": {"adds": bit_adds, "avg_steps": sum(bit_steps) / len(bit_steps),
                    "max_steps": max(bit_steps), "energy_ratio": bit_adds * e_add / e_dense},
    }


# ============================================================
# 3. Equivalence check for Eq. (22)/(23)
# ============================================================
@torch.no_grad()
def check_equivalence(model, max_tokens=32):
    """y_int = V_th * (s @ W^T) must equal the event-driven sum over time steps."""
    worst = {"ternary": 0.0, "bitwise": 0.0}
    for name, m in spiking_layers(model):
        if not hasattr(m.spike, "s_int"):
            continue
        # slice down so the spike tensor (T x ...) stays small when outliers exist
        # flatten to (tokens, in): attention/FFN inputs are 3-D, MoE expert inputs are 2-D
        s = m.spike.s_int.reshape(-1, m.spike.s_int.shape[-1])[:max_tokens].round()
        vth = m.spike.vth.reshape(-1, 1)[:max_tokens]
        W = m.lin.weight
        y_ref = vth * (s @ W.T)
        scale = y_ref.abs().max().clamp_min(1e-8)
        for key, enc, bw in [("ternary", ternary_encode, False), ("bitwise", bitwise_encode, True)]:
            y = event_driven_linear(enc(s), W, vth, bitwise=bw)
            err = ((y - y_ref).abs().max() / scale).item()
            worst[key] = max(worst[key], err)
            if err > 1e-4:
                print(f"  [!] {key} mismatch at {name}: rel_err={err:.2e}")
    return worst


# ============================================================
# 4. MoE: token distribution across experts
# ============================================================
def expert_load(model):
    loads = {}
    for name, m in model.named_modules():
        if isinstance(m, SparseMoE) and hasattr(m, "last_load"):
            l = m.last_load.float()
            loads[name] = {"share": (l / l.sum()).round(decimals=3).tolist(),
                           "dead_experts": int((l == 0).sum())}
    return loads or None


# ============================================================
# 5. Text generation
# ============================================================
@torch.no_grad()
def generate(model, n_new, block_size, device, temperature=0.8, start_id=0):
    model.eval()
    idx = torch.full((1, 1), start_id, dtype=torch.long, device=device)
    for _ in range(n_new):
        logits, _ = model(idx[:, -block_size:])
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx[0].tolist()


# ============================================================
def evaluate(model, cfg, label, val_data, meta, args, device):
    bs = cfg.block_size
    res = {"label": label, "attn_modes": cfg.attn_modes, "moe_layers": cfg.moe_layers,
           "spike_k": cfg.spike_k}
    # same val batches for every model -> differences come from the model, not sampling noise
    seed_all(args.seed)
    res["val_loss"] = estimate_loss(model, val_data, args.batch_size, bs, device, args.eval_iters)

    probe(model, val_data, args.probe_batch, bs, device)
    res["spike"] = spike_stats(model)
    res["coding"] = coding_cost(model, args.e_mac, args.e_add)
    res["moe"] = expert_load(model)
    if args.check_equiv and res["spike"]:
        res["equiv_rel_err"] = check_equivalence(model)
    if args.sample:
        res["sample"] = meta["decode"](generate(model, args.sample, bs, device,
                                                start_id=meta.get("eos_id", 0)))
    return res


def print_row(r):
    line = f"{r['label']:<28} val={r['val_loss']:.4f}"
    if r["spike"]:
        s = r["spike"]
        line += (f" | zero={s['sparsity']:.1%} spk/act={s['avg_spikes']:.2f}"
                 f" in±7={s['in_pm7']:.1%} max|s|={s['max_abs_count']}")
    if r["coding"]:
        t, b = r["coding"]["ternary"], r["coding"]["bitwise"]
        line += (f" | tern: T̄={t['avg_steps']:.1f} E={t['energy_ratio']:.2%}"
                 f" | bit: T̄={b['avg_steps']:.1f} E={b['energy_ratio']:.2%}")
    if r.get("equiv_rel_err"):
        e = r["equiv_rel_err"]
        line += f" | equiv err tern={e['ternary']:.1e} bit={e['bitwise']:.1e}"
    print(line)
    if r["moe"]:
        for name, v in r["moe"].items():
            print(f"    {name}: share={v['share']} dead={v['dead_experts']}")
    if r.get("sample"):
        print("    --- sample ---\n" + r["sample"] + "\n    --------------")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", default="data/fineweb_edu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_iters", type=int, default=100)
    ap.add_argument("--probe_batch", type=int, default=8)
    ap.add_argument("--spike_sweep", type=float, nargs="*", default=None,
                    help="spike AFTER training (no fine-tuning) with these k values")
    ap.add_argument("--check_equiv", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seed for val batch sampling")
    ap.add_argument("--sample", type=int, default=0, help="number of characters to generate")
    ap.add_argument("--e_mac", type=float, default=1.5, help="pJ per FP16 MAC (assumption)")
    ap.add_argument("--e_add", type=float, default=0.03, help="pJ per INT8 addition (assumption)")
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    _, val_data, meta = load_data(args.data)

    results = []
    for path in args.ckpts:
        name = os.path.splitext(os.path.basename(path))[0]
        if args.spike_sweep:
            for k in args.spike_sweep:
                model, cfg = load_ckpt(path, device)      # reload each time to avoid double-wrapping
                spikify(model, k)
                cfg.spike_k = k
                r = evaluate(model, cfg, f"{name}+spike(k={k})", val_data, meta, args, device)
                print_row(r); results.append(r)
        else:
            model, cfg = load_ckpt(path, device)
            r = evaluate(model, cfg, name, val_data, meta, args, device)
            print_row(r); results.append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()