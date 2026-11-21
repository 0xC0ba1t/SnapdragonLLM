"""torch.compile helpers: compile per block (regional) and unwrap compiled/DDP modules."""
from __future__ import annotations

import warnings

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

    Returns 0 and compiles nothing when ``act_ckpt == "sac"``: selective checkpointing decides what to
    save by watching individual ops in the eager forward and replays them in backward, but a compiled
    block hides its ops behind fused kernels (or, where inductor skips a dtype, runs a differently
    ordered graph), so the recompute fails with "invocation index ... not found in storage". Plain
    ``selective`` / ``full`` checkpointing re-run the whole block and are fine with compile.
    """
    cfg = getattr(getattr(model, "transformer", model), "cfg", None)
    if cfg is not None and cfg.act_ckpt == "sac":
        warnings.warn(
            "torch.compile is skipped for model.act_ckpt='sac' (selective checkpointing is incompatible with "
            "compiled blocks); use act_ckpt='full' or 'selective' if you want compile.", stacklevel=2,
        )
        return 0
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
