"""Optimizer-state memory: measure it, and predict it before allocating anything."""
from __future__ import annotations

import torch
from torch import nn

from snapdragon.config.optim import OptimConfig
from snapdragon.optim.param_groups import split_params
from snapdragon.optim.quant import DEFAULT_BLOCK

MIN_8BIT = 4096  # AdamW8bit default: smaller tensors keep exact fp32 state
BUFFER_BYTES = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}


def state_bytes(optimizer) -> int:
    """Actual bytes held in optimizer state tensors (works for HybridOptimizer too)."""
    opts = optimizer.optimizers.values() if hasattr(optimizer, "optimizers") else [optimizer]
    total = 0
    for opt in opts:
        for st in opt.state.values():
            total += sum(t.numel() * t.element_size() for t in st.values() if torch.is_tensor(t))
    return total


def estimate_state_bytes(model: nn.Module, cfg: OptimConfig, buffer_dtype: torch.dtype = torch.bfloat16) -> dict[str, int]:
    """Predicted state bytes per optimizer for the config (no allocation, meta models are fine)."""
    split = split_params(model)
    out = {"muon": sum(p.numel() for p in split.muon) * BUFFER_BYTES[buffer_dtype]}
    adam = 0
    for p in split.adam_decay + split.adam_no_decay:
        n = p.numel()
        if cfg.eight_bit and n >= MIN_8BIT:
            adam += 2 * n + 2 * 4 * -(-n // DEFAULT_BLOCK)  # two int8 codes + two fp32 scales per block
        else:
            adam += 8 * n  # fp32 exp_avg + exp_avg_sq
    out["adam"] = adam
    out["total"] = out["muon"] + out["adam"]
    return out
