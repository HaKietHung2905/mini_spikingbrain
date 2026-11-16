"""
spike_coding.py — Spike Coding (Step 2): expand integer spike counts into spike trains

  ternary_encode       : s -> T steps of {-1, 0, 1}, T = max|s|              Eq. (22)
  bitwise_encode       : bidirectional bitwise, step t carries weight 2^t     Eq. (23)
  event_driven_linear  : y = V_th · Σ_t w_t · (s_t @ W^T), identical to V_th · (s @ W^T)
"""
import torch


def ternary_encode(s: torch.Tensor) -> torch.Tensor:
    """s: integer-valued tensor -> (T, *s.shape) with values in {-1, 0, 1}."""
    mag = s.round().abs()
    T = max(int(mag.max()), 1)
    t = torch.arange(T, device=s.device).view(-1, *[1] * s.dim())
    return torch.sign(s) * (t < mag).to(s.dtype)


def bitwise_encode(s: torch.Tensor) -> torch.Tensor:
    """Bidirectional bitwise: bits of |s| as spikes, sign carried by ±1. Step t has weight 2^t."""
    mag = s.round().abs().long()
    T = max(int(mag.max()).bit_length(), 1)
    bits = torch.stack([(mag >> t) & 1 for t in range(T)]).to(s.dtype)
    return torch.sign(s) * bits


def event_driven_linear(spikes: torch.Tensor, W: torch.Tensor, vth: torch.Tensor, bitwise=False):
    """
    spikes: (T, ..., in) in {-1,0,1}; W: (out, in); vth broadcastable to (..., out).
    With s_t in {-1,0,1}, s_t @ W^T only adds / subtracts rows of W^T (no multiplications
    on event-driven hardware); 2^t is a bit shift.
    """
    y = 0
    for t, s_t in enumerate(spikes):
        y = y + (2 ** t if bitwise else 1) * (s_t @ W.T)
    return vth * y


if __name__ == "__main__":
    # quick self-test: python spike_coding.py
    torch.manual_seed(0)
    s = torch.randint(-9, 10, (4, 16)).float()
    W = torch.randn(8, 16)
    vth = torch.rand(4, 1)
    ref = vth * (s @ W.T)
    for name, enc, bw in [("ternary", ternary_encode, False), ("bitwise", bitwise_encode, True)]:
        sp = enc(s)
        y = event_driven_linear(sp, W, vth, bw)
        print(f"{name:8s} steps={sp.shape[0]:2d} spikes={int(sp.abs().sum()):4d} "
              f"max_err={(y - ref).abs().max().item():.2e}")