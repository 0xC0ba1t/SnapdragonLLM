"""Blockwise 8-bit quantization for optimizer state.

A tensor is flattened and cut into blocks; each block is scaled by its own absolute maximum (so a
single outlier only hurts its own block) and every scaled value is replaced by the index of the
nearest entry of a 256-value codebook. Storage: 1 byte per value + 4 bytes per block.
"""
from __future__ import annotations

import torch

DEFAULT_BLOCK = 2048


def linear_codebook(signed: bool = True) -> torch.Tensor:
    """256 evenly spaced values in [-1, 1] (signed) or [0, 1]."""
    if signed:
        cb = torch.linspace(-1.0, 1.0, 255)  # odd count: contains exactly 0
        return torch.cat([cb, torch.tensor([1.0])])  # pad to 256 (duplicate end point is harmless)
    return torch.linspace(0.0, 1.0, 256)


def _midpoints(codebook: torch.Tensor) -> torch.Tensor:
    return (codebook[1:] + codebook[:-1]) / 2


def quantize_blockwise(
    x: torch.Tensor, codebook: torch.Tensor, block: int = DEFAULT_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (codes uint8 of shape (numel,), absmax float32 of shape (n_blocks,))."""
    flat = x.detach().reshape(-1).float()
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(dim=1)
    scaled = blocks / absmax.clamp(min=torch.finfo(torch.float32).tiny).unsqueeze(1)
    codes = torch.bucketize(scaled, _midpoints(codebook.to(x.device)))
    return codes.reshape(-1)[:n].to(torch.uint8), absmax


def dequantize_blockwise(
    codes: torch.Tensor,
    absmax: torch.Tensor,
    codebook: torch.Tensor,
    shape: torch.Size | tuple,
    block: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    n = codes.numel()
    pad = (-n) % block
    vals = codebook.to(codes.device)[codes.long()]
    if pad:
        vals = torch.cat([vals, vals.new_zeros(pad)])
    out = (vals.view(-1, block) * absmax.unsqueeze(1)).reshape(-1)[:n]
    return out.view(shape)
