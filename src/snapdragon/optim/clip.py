"""Global gradient-norm clipping that leaves non-finite gradients untouched for the caller."""
from __future__ import annotations

from collections.abc import Iterable

import torch


@torch.no_grad()
def clip_grad_norm(params: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor:
    """Scale gradients so their global L2 norm is at most ``max_norm`` (0 disables clipping).

    Returns the pre-clip norm as a float32 tensor on the gradients' device (no host sync). If the
    norm is NaN/inf the gradients are NOT modified: the trainer checks ``torch.isfinite(norm)`` and
    skips the step instead of poisoning the weights with a clipped NaN.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    norms = torch._foreach_norm(grads)
    total = torch.linalg.vector_norm(torch.stack([n.float() for n in norms]))
    if max_norm > 0:
        # branch-free: when the norm is non-finite the coefficient is 1 and nothing changes
        coef = torch.where(torch.isfinite(total), (max_norm / (total + 1e-6)).clamp(max=1.0), torch.ones_like(total))
        torch._foreach_mul_(grads, coef.to(grads[0].dtype) if grads[0].dtype != torch.float64 else coef)
    return total
