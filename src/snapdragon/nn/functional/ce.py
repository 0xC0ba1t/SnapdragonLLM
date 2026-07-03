"""Cross-entropy with z-loss: a plain reference and a memory-lean chunked version."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def _logits(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Logits upcast to at least float32 (float64 inputs stay float64, which the tests rely on)."""
    out = h @ w.T
    return out.to(torch.promote_types(out.dtype, torch.float32))


def reference_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    z_loss_coeff: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Materialises all logits. loss = mean CE + z_loss_coeff * mean(logsumexp^2) over valid tokens."""
    logits = _logits(hidden, weight)
    valid = targets != ignore_index
    n = valid.sum().clamp(min=1)
    ce = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="sum") / n
    if z_loss_coeff:
        lse = torch.logsumexp(logits, dim=-1)
        ce = ce + z_loss_coeff * (lse.pow(2) * valid).sum() / n
    return ce


def _chunk_sums(h, w, t, z_loss_coeff, ignore_index):
    logits = _logits(h, w)
    valid = t != ignore_index
    ce = F.cross_entropy(logits, t, ignore_index=ignore_index, reduction="sum")
    if z_loss_coeff:
        ce = ce + z_loss_coeff * (torch.logsumexp(logits, dim=-1).pow(2) * valid).sum()
    return ce


def chunked_ce_checkpointed(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    z_loss_coeff: float = 0.0,
    chunk_tokens: int = 4096,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Same value as ``reference_ce`` but the logits of each chunk are recomputed in backward
    (activation checkpointing), so at most ``chunk_tokens x vocab`` logits exist at any time."""
    n = (targets != ignore_index).sum().clamp(min=1)
    total = hidden.new_zeros((), dtype=torch.promote_types(hidden.dtype, torch.float32))
    for start in range(0, hidden.shape[0], chunk_tokens):
        sl = slice(start, start + chunk_tokens)
        total = total + checkpoint(
            _chunk_sums, hidden[sl], weight, targets[sl], z_loss_coeff, ignore_index, use_reentrant=False
        )
    return total / n
