"""Language-model loss head: tied/untied output matrix + chunked CE + z-loss, with a breakdown."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from snapdragon.nn.functional.ce import IGNORE_INDEX, chunked_ce


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
    """hidden (B, T, d), targets (B, T). The CE/z split for logging comes from two cheap
    no-grad passes only when needed, so the training path stays a single fused call."""
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    loss = chunked_ce(h, weight, t, z_loss_coeff, chunk_tokens, ignore_index)
    with torch.no_grad():
        total_with_z = loss.detach()
        ce_only = chunked_ce(h.detach(), weight.detach(), t, 0.0, chunk_tokens, ignore_index)
        z = (total_with_z - ce_only) / z_loss_coeff if z_loss_coeff else torch.zeros_like(ce_only)
    return LossOutput(loss=loss, ce=ce_only, z=z)
