"""torch.compile helpers: compile per block (regional) and unwrap compiled/DDP modules."""
from __future__ import annotations

from torch import nn

from snapdragon.utils.device import compile_supported


def should_compile(setting: str) -> bool:
    """``setting`` is the config value: "auto" (CUDA on Linux only), "true" or "false"."""
    if setting == "auto":
        return compile_supported()
    if setting in ("true", "false"):
        return setting == "true"
    raise ValueError(f"compile must be auto, true or false, got {setting!r}")


def compile_blocks(model: nn.Module, backend: str = "inductor", **kwargs) -> int:
    """Compile every transformer block in place; returns how many were compiled.

    Regional compilation compiles one block's graph and reuses it for all identical layers, so
    compile time is ~1/n_layers of compiling the whole model and recompiles stay local. Blocks are
    wrapped via ``nn.Module.compile`` so state_dict keys do not change.
    """
    blocks = model.transformer.blocks if hasattr(model, "transformer") else model.blocks
    for block in blocks:
        block.compile(backend=backend, **kwargs)
    return len(blocks)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP/FSDP and torch.compile wrappers to reach the underlying module."""
    while True:
        if hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model
