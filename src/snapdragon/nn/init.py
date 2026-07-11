"""Weight initialisation."""
from __future__ import annotations

import math

import torch
from torch import nn

from snapdragon.config.model import ModelConfig
from snapdragon.nn.norm import RMSNorm


def _normal_(t: torch.Tensor, std: float, gen: torch.Generator) -> None:
    """Fill t ~ N(0, std^2) from a CPU generator, whatever device t lives on."""
    with torch.no_grad():
        t.copy_(torch.empty(t.shape, dtype=torch.float32).normal_(0.0, std, generator=gen))


def init_weights(model: nn.Module, cfg: ModelConfig) -> None:
    """Normal(0, init_std) everywhere; layers feeding the residual stream (marked
    ``is_residual_output``) are scaled by 1/sqrt(2 * n_layers) so the stream's variance does not
    grow with depth (GPT-2 / Megatron style). Norm gains start at 1.

    Draws come from a private generator seeded with ``cfg.init_seed`` in module order, so the
    result is identical for a direct build, a meta-device build + materialize, and any device.
    """
    gen = torch.Generator().manual_seed(cfg.init_seed)
    residual_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            std = residual_std if getattr(module, "is_residual_output", False) else cfg.init_std
            _normal_(module.weight, std, gen)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            _normal_(module.weight, cfg.init_std, gen)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)  # explicit, so re-init after to_empty restores the gains
