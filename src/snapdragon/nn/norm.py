"""RMSNorm."""
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """y = x / rms(x) * weight, computed in float32 and cast back to the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
