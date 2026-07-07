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


class _ChunkedCE(torch.autograd.Function):
    """Fused linear + cross-entropy + z-loss that never keeps more than one chunk of logits.

    forward  : per chunk compute logits, logsumexp and the target logit; save only the (N,)
               logsumexp vector, never the (N, V) logits.
    backward : per chunk recompute logits, form dL/dlogits = (softmax * (1 + 2 z lse) - onehot)
               * valid * g / n and push it through the matmul to get dL/dh and dL/dW.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, z_loss_coeff, chunk_tokens, ignore_index):
        w = weight.to(hidden.dtype)
        valid = targets != ignore_index
        n_valid = valid.sum().clamp(min=1)
        acc = torch.promote_types(hidden.dtype, torch.float32)
        lse_all = torch.empty(hidden.shape[0], dtype=acc, device=hidden.device)
        ce_sum = torch.zeros((), dtype=acc, device=hidden.device)
        z_sum = torch.zeros((), dtype=acc, device=hidden.device)
        for start in range(0, hidden.shape[0], chunk_tokens):
            sl = slice(start, start + chunk_tokens)
            logits = _logits(hidden[sl], w)
            lse = torch.logsumexp(logits, dim=-1)
            lse_all[sl] = lse
            tgt = logits.gather(1, targets[sl].clamp(min=0).unsqueeze(1)).squeeze(1)
            v = valid[sl]
            ce_sum = ce_sum + ((lse - tgt) * v).sum()
            z_sum = z_sum + (lse.pow(2) * v).sum()
        ctx.save_for_backward(hidden, weight, targets, lse_all, n_valid)
        ctx.z, ctx.chunk, ctx.ignore = z_loss_coeff, chunk_tokens, ignore_index
        ce, z = ce_sum / n_valid, z_sum / n_valid
        ctx.mark_non_differentiable(ce, z)
        return ce + z_loss_coeff * z, ce, z

    @staticmethod
    def backward(ctx, grad_out, _grad_ce, _grad_z):
        hidden, weight, targets, lse_all, n_valid = ctx.saved_tensors
        w = weight.to(hidden.dtype)
        valid = targets != ctx.ignore
        grad_h = torch.empty_like(hidden)
        grad_w = torch.zeros(weight.shape, dtype=lse_all.dtype, device=weight.device)
        scale = grad_out / n_valid
        for start in range(0, hidden.shape[0], ctx.chunk):
            sl = slice(start, start + ctx.chunk)
            h = hidden[sl]
            logits = _logits(h, w)
            lse = lse_all[sl]
            g = torch.exp(logits - lse.unsqueeze(1)) * (1 + 2 * ctx.z * lse).unsqueeze(1)
            g.scatter_add_(1, targets[sl].clamp(min=0).unsqueeze(1), -torch.ones_like(g[:, :1]))
            g = g * (valid[sl].unsqueeze(1) * scale)
            g = g.to(hidden.dtype)
            grad_h[sl] = g @ w
            grad_w += (g.T @ h).to(grad_w.dtype)
        return grad_h, grad_w.to(weight.dtype), None, None, None, None


def chunked_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    z_loss_coeff: float = 0.0,
    chunk_tokens: int = 4096,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Memory-lean fused linear + cross-entropy (+ z-loss). hidden (N, d), weight (V, d), targets (N,)."""
    return _ChunkedCE.apply(hidden, weight, targets, z_loss_coeff, chunk_tokens, ignore_index)[0]


def chunked_ce_with_stats(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    z_loss_coeff: float = 0.0,
    chunk_tokens: int = 4096,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like ``chunked_ce`` but also returns the (non-differentiable) mean CE and mean logsumexp^2
    as by-products of the same pass, so logging them costs nothing extra."""
    return _ChunkedCE.apply(hidden, weight, targets, z_loss_coeff, chunk_tokens, ignore_index)
