"""Weight initialisation."""
from __future__ import annotations

import math

from torch import nn

from snapdragon.config.model import ModelConfig


def init_weights(model: nn.Module, cfg: ModelConfig) -> None:
    """Normal(0, init_std) everywhere; layers feeding the residual stream (marked
    ``is_residual_output``) are scaled by 1/sqrt(2 * n_layers) so the stream's variance does not
    grow with depth (GPT-2 / Megatron style). Norm gains start at 1."""
    residual_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            std = residual_std if getattr(module, "is_residual_output", False) else cfg.init_std
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=cfg.init_std)
