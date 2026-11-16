"""
model.py — GPT backbone configurable per layer (attention type, dense FFN or MoE)

attn_modes[i] in:
  "fa" | "swa" | "la"          -> Attention(mode=...)
  "hybrid_swa" | "hybrid_fa"   -> HybridAttention(softmax_mode="swa" | "fa")
Layers listed in moe_layers use SparseMoE instead of SwiGLU.
spike_k is not used here: train.build_model() wraps Linear layers with SpikingLinear.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import RMSNorm, SwiGLU
from attention import Attention, HybridAttention
from moe import SparseMoE


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 512
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    ffn_hidden: int = 1024
    window: int = 128
    attn_modes: list = field(default_factory=lambda: ["fa"] * 6)
    moe_layers: list = field(default_factory=list)
    n_experts: int = 8
    top_k: int = 1
    n_shared: int = 1
    spike_k: Optional[float] = None
    moe_aux_coef: float = 0.01

    def __post_init__(self):
        assert len(self.attn_modes) == self.n_layer, "attn_modes must have one entry per layer"


def make_attention(cfg: GPTConfig, mode: str) -> nn.Module:
    if mode in ("fa", "swa", "la"):
        return Attention(cfg.d_model, cfg.n_head, mode=mode, window=cfg.window)
    if mode in ("hybrid_swa", "hybrid_fa"):
        return HybridAttention(cfg.d_model, cfg.n_head, softmax_mode=mode.split("_")[1],
                               window=cfg.window)
    raise ValueError(f"unknown attention mode: {mode}")


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, i: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = make_attention(cfg, cfg.attn_modes[i])
        self.norm2 = RMSNorm(cfg.d_model)
        if i in cfg.moe_layers:
            self.ffn = SparseMoE(cfg.d_model, cfg.n_experts, cfg.top_k,
                                 lambda: SwiGLU(cfg.d_model, cfg.ffn_hidden), cfg.n_shared)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)   # learned absolute positions
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.wte.weight                     # weight tying: saves vocab x d params

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm_f(x))

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if self.training and self.cfg.moe_layers:
            aux = sum(m.aux_loss for m in self.modules() if isinstance(m, SparseMoE))
            loss = loss + self.cfg.moe_aux_coef * aux
        return logits, loss