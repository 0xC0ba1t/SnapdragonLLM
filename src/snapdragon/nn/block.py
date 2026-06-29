"""One pre-norm transformer block."""
from __future__ import annotations

import torch
from torch import nn

from snapdragon.config.model import ModelConfig
from snapdragon.nn.attention import Attention
from snapdragon.nn.mlp import SwiGLU
from snapdragon.nn.norm import RMSNorm
from snapdragon.nn.rope import RotaryEmbedding


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def forward(
        self, x: torch.Tensor, rope: RotaryEmbedding, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope, position_ids)
        return x + self.mlp(self.mlp_norm(x))
