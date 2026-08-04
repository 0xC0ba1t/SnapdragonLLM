"""Muon: momentum SGD whose update for each weight matrix is (approximately) orthogonalised.

Orthogonalising the update equalises its singular values, so every direction in weight space gets
a comparable step. It is cheap (a few small matmuls per matrix) and needs only ONE momentum buffer
per parameter, half the optimizer-state memory of AdamW.
"""
from __future__ import annotations

import torch

# Quintic Newton-Schulz coefficients (maximise the slope at zero; the iteration lands singular values
# in roughly [0.7, 1.2] instead of converging exactly to 1, which is fine for optimisation).
NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate U V^T from the SVD g = U S V^T using ``steps`` quintic iterations in bf16.

    Accepts a matrix (rows, cols) or a stack (n, rows, cols); every matrix is normalised and
    iterated independently, so a stack costs one batched matmul per step instead of n small ones.
    """
    if g.ndim not in (2, 3):
        raise ValueError(f"newton_schulz expects a matrix or a stack of matrices, got shape {tuple(g.shape)}")
    a, b, c = NS_COEFFS
    x = g.to(torch.bfloat16)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    # spectral norm <= Frobenius norm <= 1, as the iteration requires
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.mT if transposed else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2-D parameters only. Update: nesterov momentum -> Newton-Schulz -> shape scale,
    then decoupled weight decay. The momentum buffer is kept in bf16 to halve its memory."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
        buffer_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                        ns_steps=ns_steps, buffer_dtype=buffer_dtype)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon only handles 2-D parameters, got shape {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=group["buffer_dtype"])
                buf = state["momentum_buffer"]
                buf.lerp_(g.to(buf.dtype), 1 - mom)  # buf = mom * buf + (1 - mom) * g
                update = g.lerp(buf.to(g.dtype), mom) if group["nesterov"] else buf.to(g.dtype)
                update = newton_schulz(update, group["ns_steps"])
                rows, cols = p.shape
                update = update * max(1.0, rows / cols) ** 0.5  # tall matrices get proportionally larger steps
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        # torch casts floating state to the parameter dtype on load, which would silently turn the
        # bf16 momentum buffers into fp32 (2x memory, different rounding) after every resume
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st and "momentum_buffer" in st:
                    st["momentum_buffer"] = st["momentum_buffer"].to(group["buffer_dtype"])
