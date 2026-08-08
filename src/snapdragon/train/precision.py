"""Mixed-precision context for the forward pass."""
from __future__ import annotations

import contextlib

import torch

from snapdragon.utils.device import autocast_dtype


def autocast_context(precision: str, device: torch.device):
    """bf16 autocast on CUDA/CPU (matmuls in bf16, norms/softmax stay fp32); fp32 means no autocast.

    Master weights and gradients are always fp32; autocast only changes compute dtype.
    """
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type not in ("cuda", "cpu"):
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=autocast_dtype(precision))
