"""Rotary position embeddings (half-split convention, as in Llama)."""
from __future__ import annotations

import torch
from torch import nn


def rope_tables(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_seq_len, head_dim // 2), in float32."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[..., i], x[..., i + d/2]). ``cos``/``sin`` broadcast against (..., T, d/2)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return out.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 500_000.0) -> None:
        super().__init__()
        self.head_dim, self.max_seq_len, self.theta = head_dim, max_seq_len, theta
        cos, sin = rope_tables(head_dim, max_seq_len, theta)
        # not persistent: cheap to rebuild, and keeps checkpoints independent of max_seq_len/theta
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def rebuild(self, device: torch.device | str | None = None) -> None:
        """Recompute the tables in place. Needed after ``to_empty`` (meta -> real), which leaves
        buffers uninitialised."""
        cos, sin = rope_tables(self.head_dim, self.max_seq_len, self.theta)
        device = device or self.cos.device
        self.cos = cos.to(device)
        self.sin = sin.to(device)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q: (B, Hq, T, d), k: (B, Hkv, T, d); position_ids: optional (B, T)."""
        T = q.shape[-2]
        if position_ids is None:
            cos, sin = self.cos[:T], self.sin[:T]  # (T, d/2) broadcasts over B, H
        else:
            cos = self.cos[position_ids].unsqueeze(1)  # (B, 1, T, d/2)
            sin = self.sin[position_ids].unsqueeze(1)
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
