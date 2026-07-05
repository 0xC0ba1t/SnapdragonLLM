"""Choose which scaled-dot-product-attention kernels PyTorch may use."""
from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_BACKENDS = {
    "flash": [SDPBackend.FLASH_ATTENTION],
    "efficient": [SDPBackend.EFFICIENT_ATTENTION],
    "math": [SDPBackend.MATH],
    "auto": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
}


def resolve_backend(name: str | None = None) -> str:
    """Explicit name, else $SNAPDRAGON_SDPA, else 'auto'."""
    name = name or os.environ.get("SNAPDRAGON_SDPA", "auto")
    if name not in _BACKENDS:
        raise ValueError(f"unknown sdpa backend {name!r}; choose from {sorted(_BACKENDS)}")
    return name


@contextlib.contextmanager
def sdpa_backend(name: str | None = None) -> Iterator[str]:
    """Restrict SDPA to the named backend(s). 'auto' lets PyTorch pick flash -> efficient -> math,
    which is what you want on CUDA; on CPU only the math path exists and is used automatically."""
    name = resolve_backend(name)
    if name == "auto" or not torch.cuda.is_available():
        yield name
        return
    with sdpa_kernel(_BACKENDS[name]):
        yield name
