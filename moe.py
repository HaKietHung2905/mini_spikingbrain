"""
moe.py — Sparse MoE and upcycling (Sec. 3.2)

  SparseMoE : sigmoid router -> top-k routed experts + always-on shared experts   Eq. (16)(17)
  upcycle   : copy a dense SwiGLU into N experts, scale all 3 matrices by
              alpha = (1 / (S + k/N))^(1/3) so the MoE output ≈ dense output   Eq. (15)(18)
"""
import copy

import torch
import torch.nn as nn


class SparseMoE(nn.Module):
    def __init__(self, d, n_exp, top_k, ffn_factory, n_shared=1):
        super().__init__()
        self.router = nn.Linear(d, n_exp, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)        # random router, p_i ≈ 0.5 at start
        self.experts = nn.ModuleList(ffn_factory() for _ in range(n_exp))
        self.shared = nn.ModuleList(ffn_factory() for _ in range(n_shared))
        self.n_exp, self.k = n_exp, top_k
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        x2 = x.reshape(-1, D)
        p = torch.sigmoid(self.router(x2))                     # (N, n_exp)
        w, idx = p.topk(self.k, dim=-1)                        # (N, k)

        out = sum(e(x2) for e in self.shared) if len(self.shared) else torch.zeros_like(x2)
        for j, expert in enumerate(self.experts):
            hit = idx == j                                     # (N, k)
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            gate = (w * hit)[rows].sum(-1, keepdim=True)       # p_j for the tokens routed to j
            out = out.index_add(0, rows, gate * expert(x2[rows]))

        # Switch-style load-balancing loss: n_exp * sum_i (fraction routed to i) * (mean prob of i)
        load = torch.bincount(idx.flatten(), minlength=self.n_exp).float()
        frac = load / load.sum()
        prob = (p / p.sum(-1, keepdim=True)).mean(0)
        self.aux_loss = self.n_exp * (frac.detach() * prob).sum()
        self.last_load = load.detach()
        return out.view(B, T, D)


@torch.no_grad()
def upcycle(dense_ffn, d, n_exp=8, top_k=1, n_shared=1, use_alpha=True):
    alpha = (1.0 / (n_shared + top_k / n_exp)) ** (1.0 / 3.0) if use_alpha else 1.0

    def make():
        e = copy.deepcopy(dense_ffn)
        for lin in (e.w1, e.w2, e.w3):                         # SwiGLU: 3 matrices -> alpha^3 overall
            lin.weight.mul_(alpha)
        return e

    moe = SparseMoE(d, n_exp, top_k, make, n_shared)
    print(f"[upcycle] alpha={alpha:.4f} (S={n_shared}, k={top_k}, N={n_exp})")
    return moe