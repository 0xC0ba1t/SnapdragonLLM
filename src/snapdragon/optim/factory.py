"""Build the optimizer described by an OptimConfig for a model."""
from __future__ import annotations

import torch
from torch import nn

from snapdragon.config.optim import OptimConfig
from snapdragon.optim.adamw8bit import AdamW8bit
from snapdragon.optim.hybrid import HybridOptimizer
from snapdragon.optim.muon import Muon
from snapdragon.optim.param_groups import assert_partition, split_params


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> HybridOptimizer:
    """Muon on the hidden matrices; AdamW (8-bit unless ``cfg.eight_bit`` is off) on the rest."""
    state_device = "cpu" if cfg.offload_states else None
    split = split_params(model)
    assert_partition(model, split)
    optimizers: dict[str, torch.optim.Optimizer] = {}
    if split.muon:
        optimizers["muon"] = Muon(
            split.muon, lr=cfg.muon_lr, momentum=cfg.muon_momentum, weight_decay=cfg.muon_weight_decay,
            state_device=state_device,
        )
    adam_groups = []
    if split.adam_decay:
        adam_groups.append({"params": split.adam_decay, "weight_decay": cfg.adam_weight_decay})
    if split.adam_no_decay:
        adam_groups.append({"params": split.adam_no_decay, "weight_decay": 0.0})
    if adam_groups:
        if cfg.eight_bit:
            optimizers["adam"] = AdamW8bit(
                adam_groups, lr=cfg.adam_lr, betas=cfg.adam_betas, eps=cfg.adam_eps, state_device=state_device
            )
        else:
            fused = torch.cuda.is_available() and all(p.is_cuda for g in adam_groups for p in g["params"])
            optimizers["adam"] = torch.optim.AdamW(
                adam_groups, lr=cfg.adam_lr, betas=cfg.adam_betas, eps=cfg.adam_eps, fused=fused or None
            )
    return HybridOptimizer(**optimizers)
