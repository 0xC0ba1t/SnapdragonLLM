"""Build models from configs, optionally on the meta device (no memory) first."""
from __future__ import annotations

import torch

from snapdragon.config.model import ModelConfig
from snapdragon.config.schema import Config
from snapdragon.model.lm import CausalLM


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters; tied weights are counted once (``parameters()`` de-duplicates)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def build_model(
    cfg: Config | ModelConfig,
    z_loss_coeff: float | None = None,
    ce_chunk_tokens: int | None = None,
    device: torch.device | str | None = None,
) -> CausalLM:
    """Construct a CausalLM. Pass ``device="meta"`` to get shapes only (instant, zero memory)."""
    if isinstance(cfg, Config):
        model_cfg = cfg.model
        z = cfg.train.z_loss_coeff if z_loss_coeff is None else z_loss_coeff
        chunk = cfg.train.ce_chunk_tokens if ce_chunk_tokens is None else ce_chunk_tokens
    else:
        model_cfg = cfg
        z = 1e-4 if z_loss_coeff is None else z_loss_coeff
        chunk = 4096 if ce_chunk_tokens is None else ce_chunk_tokens
    with torch.device(device) if device is not None else _nullcontext():
        return CausalLM(model_cfg, z_loss_coeff=z, ce_chunk_tokens=chunk)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
