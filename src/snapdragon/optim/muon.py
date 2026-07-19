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
    """Approximate U V^T from the SVD g = U S V^T using ``steps`` quintic iterations in bf16."""
    if g.ndim != 2:
        raise ValueError(f"newton_schulz expects a matrix, got shape {tuple(g.shape)}")
    a, b, c = NS_COEFFS
    x = g.to(torch.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)  # spectral norm <= Frobenius norm <= 1, as the iteration requires
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.T if transposed else x).to(g.dtype)
