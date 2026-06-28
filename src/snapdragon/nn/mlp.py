"""SwiGLU feed-forward block."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """down( silu(gate(x)) * up(x) ) with gate and up fused into one matmul."""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.gate_up_proj = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.down_proj.is_residual_output = True  # picked up by init to scale by 1/sqrt(2 L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
