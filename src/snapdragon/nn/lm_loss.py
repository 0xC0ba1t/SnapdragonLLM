"""Language-model loss head: tied/untied output matrix + chunked CE + z-loss, with a breakdown."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from snapdragon.nn.functional.ce import IGNORE_INDEX, chunked_ce_with_stats


@dataclass
class LossOutput:
    loss: torch.Tensor  # what to backprop: ce + z_loss_coeff * z
    ce: torch.Tensor  # detached, for logging
    z: torch.Tensor  # detached mean of logsumexp^2 (before the coefficient)


def lm_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    z_loss_coeff: float = 1e-4,
    chunk_tokens: int = 4096,
    ignore_index: int = IGNORE_INDEX,
) -> LossOutput:
    """hidden (B, T, d), targets (B, T). One fused pass yields the loss and its logging breakdown."""
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    loss, ce, z = chunked_ce_with_stats(h, weight, t, z_loss_coeff, chunk_tokens, ignore_index)
    return LossOutput(loss=loss, ce=ce, z=z)
