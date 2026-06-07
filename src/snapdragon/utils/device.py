"""Device and precision selection that degrades gracefully (CUDA, then MPS, then CPU)."""
from __future__ import annotations

import sys

import torch


def get_device(prefer: str | None = None) -> torch.device:
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bf16_supported(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    return device.type == "cpu"


def compile_supported() -> bool:
    """torch.compile needs Triton for GPU kernels, which is effectively Linux-only."""
    return torch.cuda.is_available() and sys.platform.startswith("linux")


_DTYPES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
    "fp32": torch.float32, "float32": torch.float32,
}


def autocast_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"unknown precision {name!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]
