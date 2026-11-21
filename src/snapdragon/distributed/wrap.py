"""Wrap a model for data-parallel (DDP) or sharded (FSDP2) training."""
from __future__ import annotations

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from snapdragon.distributed.env import DistEnv

PARALLEL_MODES = ("none", "ddp")  # fsdp.py is experimental and deliberately not selectable here


def ddp_device_ids(model: nn.Module, env: DistEnv) -> list[int] | None:
    """``device_ids`` for DDP: the rank's GPU only if the model really lives on CUDA. Asking whether CUDA
    *exists* is wrong: a CPU model on a machine that has GPUs (a gloo test, or ``--device cpu``) must get
    ``None``, otherwise DDP mixes a CUDA device with CPU parameters."""
    first = next(model.parameters(), None)
    return [env.local_rank] if first is not None and first.is_cuda else None


def wrap_ddp(model: nn.Module, env: DistEnv) -> nn.Module:
    """Replicate the model across ranks, all-reducing gradients (bucketed, overlapped with backward).

    ``find_unused_parameters`` stays False: every parameter is used every step, and detecting
    unused ones costs an extra graph traversal per iteration.
    """
    if not env.distributed:
        return model
    return DDP(model, device_ids=ddp_device_ids(model, env), find_unused_parameters=False, gradient_as_bucket_view=True)


def wrap_model(model: nn.Module, env: DistEnv, mode: str) -> nn.Module:
    if mode not in PARALLEL_MODES:
        raise ValueError(f"parallel must be one of {PARALLEL_MODES}, got {mode!r}")
    if mode == "none" or not env.distributed:
        return model
    return wrap_ddp(model, env)
