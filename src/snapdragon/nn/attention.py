"""Grouped-query causal self-attention with optional QK-norm."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from snapdragon.config.model import ModelConfig
from snapdragon.nn.norm import RMSNorm
from snapdragon.nn.rope import RotaryEmbedding


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        q_dim = cfg.n_heads * cfg.head_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.d_model, q_dim + 2 * kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.d_model, bias=False)
        self.o_proj.is_residual_output = True
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self._split = (q_dim, kv_dim, kv_dim)

    def forward(
        self, x: torch.Tensor, rope: RotaryEmbedding, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv_proj(x).split(self._split, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)  # QK-norm keeps attention logits bounded
        q, k = rope(q, k, position_ids)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.n_heads != self.n_kv_heads)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))
