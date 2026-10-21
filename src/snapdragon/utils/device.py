"""Device and precision selection that degrades gracefully (CUDA, then MPS, then CPU)."""
from __future__ import annotations

import os
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


def configure_backends(env: dict | None = None) -> dict:
    """Process-wide performance settings; call once before the first CUDA allocation.

    * TF32 matmuls for any fp32 math that remains (loss, norms) on Ampere+: ~free speedup, bf16 does the rest.
    * ``expandable_segments`` lets the CUDA caching allocator grow/shrink segments instead of keeping
      fixed-size blocks, which sharply reduces fragmentation-induced OOMs on 12-16 GB cards.
    Returns what was applied (useful for logging and tests). ``env`` defaults to ``os.environ``.
    """
    env = os.environ if env is None else env
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return {"alloc_conf": env["PYTORCH_CUDA_ALLOC_CONF"], "matmul_precision": torch.get_float32_matmul_precision()}
