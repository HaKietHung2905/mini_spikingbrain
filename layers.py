"""
layers.py — Basic building blocks

  RMSNorm        : LLaMA-style normalization
  SwiGLU         : FFN with 3 matrices (w1 gate, w3 up, w2 down) -> cube root in Eq. (18)
  AdaptiveSpike  : Adaptive Threshold Spiking (Step 1): x -> round(x / V_th(x)) * V_th(x)
  SpikingLinear  : nn.Linear whose INPUT is spiked first
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)   # gate
        self.w3 = nn.Linear(d, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, d, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AdaptiveSpike(nn.Module):
    """
    V_th(x) = mean(|x|) / k        (per token, over the last dim)
    s_INT   = round(x / V_th)      (integer spike count, signed)
    output  = s_INT * V_th         (≈ x, so pretrained weights still fit)
    Gradient passes through round() with the straight-through estimator.
    In eval mode the last s_INT and V_th are kept for statistics / spike coding.
    """
    def __init__(self, k=1.0):
        super().__init__()
        self.k = k

    def forward(self, x):
        vth = x.abs().mean(-1, keepdim=True).clamp_min(1e-6) / self.k
        s = x / vth
        s_int = s + (torch.round(s) - s).detach()          # STE
        if not self.training:
            self.s_int, self.vth = s_int.detach(), vth.detach()
        return s_int * vth

    def extra_repr(self):
        return f"k={self.k}"


class SpikingLinear(nn.Module):
    """Spike the input, then apply the original linear layer: y = V_th · (s_INT @ W^T)."""
    def __init__(self, lin: nn.Linear, k=1.0):
        super().__init__()
        self.lin = lin
        self.spike = AdaptiveSpike(k)

    def forward(self, x):
        return self.lin(self.spike(x))