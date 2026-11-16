"""
attention.py — FA / SWA / LA and the intra-layer hybrid

All variants share the same q/k/v/o parameter names, so weights from a pretrained
FA model load directly into SWA, LA or hybrid layers (Attention Map Correspondence, Sec. 3.1).

  Attention(mode="fa")   : softmax with causal mask                      Eq. (1)
  Attention(mode="swa")  : softmax with windowed causal mask             Eq. (2)
  Attention(mode="la")   : (φ(Q)φ(K)^T ⊙ M) V, φ = ReLU, + output norm   Eq. (3)
  HybridAttention        : norm(SWA or FA) + norm(LA), in parallel (SpikingBrain-76B style)
  la_recurrent           : recurrent form of LA, Eq. (4): S_t = S_{t-1} + k_t^T v_t, o_t = q_t S_t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import RMSNorm


class Attention(nn.Module):
    def __init__(self, d, n_head, mode="fa", window=128):
        super().__init__()
        assert d % n_head == 0
        assert mode in ("fa", "swa", "la")
        self.h, self.dh, self.mode, self.w = n_head, d // n_head, mode, window
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if mode == "la":
            # LA has no softmax denominator -> normalize its output (Sec. 3.1, point 1)
            self.norm_la = RMSNorm(self.dh)

    def split(self, x):                                   # (B,T,d) -> (B,h,T,dh)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def core(self, q, k, v, mode):
        T = q.size(2)
        if mode == "fa":
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if mode == "swa":
            i = torch.arange(T, device=q.device)
            mask = (i[:, None] >= i[None]) & (i[:, None] - i[None] < self.w)   # True = attend
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if mode == "la":
            q, k = F.relu(q), F.relu(k)                   # non-negative map, like softmax
            A = (q @ k.transpose(-1, -2)).tril()          # QK^T ⊙ M (0/1 mask)
            return A @ v
        raise ValueError(mode)

    def merge(self, o):                                   # (B,h,T,dh) -> (B,T,d)
        return self.o(o.transpose(1, 2).flatten(2))

    def forward(self, x):
        q, k, v = map(self.split, (self.q(x), self.k(x), self.v(x)))
        o = self.core(q, k, v, self.mode)
        if self.mode == "la":
            o = self.norm_la(o)
        return self.merge(o)


class HybridAttention(Attention):
    """Intra-layer hybrid: softmax branch (SWA or FA) + LA branch, each normalized, then summed."""
    def __init__(self, d, n_head, softmax_mode="swa", window=128):
        super().__init__(d, n_head, mode=softmax_mode, window=window)
        self.norm_sm = RMSNorm(self.dh)                   # new, small params (Sec. 3.1, point 2)
        self.norm_la = RMSNorm(self.dh)

    def forward(self, x):
        q, k, v = map(self.split, (self.q(x), self.k(x), self.v(x)))
        o = self.norm_sm(self.core(q, k, v, self.mode)) + self.norm_la(self.core(q, k, v, "la"))
        return self.merge(o)


def la_recurrent(q, k, v):
    """Recurrent LA with an O(1) state. q, k must already be passed through φ. Shapes (B,h,T,dh)."""
    S = q.new_zeros(*q.shape[:2], q.size(-1), v.size(-1))
    out = []
    for t in range(q.size(2)):
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]        # S_t = S_{t-1} + k_t^T v_t
        out.append((q[:, :, t, None, :] @ S).squeeze(-2))        # o_t = q_t S_t
    return torch.stack(out, dim=2)