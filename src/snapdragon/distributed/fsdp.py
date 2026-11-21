"""EXPERIMENTAL: FSDP2 (``fully_shard``) wrapping, not selectable from train.py yet.

Shards parameters, gradients and optimizer state across ranks. Checkpointing sharded weights needs a
collective full-state-dict gather on every rank, which the trainer does not do yet, so this module is
library code only (its multi-process test runs on Linux CI).

Use this when the model + optimizer do not fit one GPU even with checkpointing and 8-bit state.
Muon needs whole matrices and the 8-bit AdamW works on flat local tensors, so under FSDP the optimizer
falls back to the stock fused AdamW (``build_fsdp_optimizer``), which understands sharded DTensors.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from snapdragon.config.optim import OptimConfig
from snapdragon.optim.param_groups import split_params

try:  # public since torch 2.6; lived under _composable in 2.4/2.5
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
except ImportError:  # pragma: no cover
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard


def mesh_device_type(model: nn.Module) -> str:
    """The device mesh must match where the parameters are, not whether a GPU happens to exist."""
    first = next(model.parameters(), None)
    return "cuda" if first is not None and first.is_cuda else "cpu"


def wrap_fsdp2(model: nn.Module, param_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Shard each transformer block, then the root (embeddings + final norm).

    Parameters are all-gathered per block just in time, computed in ``param_dtype`` and gradients are
    reduce-scattered in fp32. Sharding per block keeps at most one block's full weights alive.
    """
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    mesh = init_device_mesh(mesh_device_type(model), (dist.get_world_size(),))
    transformer = model.transformer if hasattr(model, "transformer") else model
    for block in transformer.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    return model


def build_fsdp_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.AdamW:
    """Stock AdamW over decayed / non-decayed groups (all parameters, including the matrices)."""
    split = split_params(model)
    decay = split.muon + split.adam_decay
    groups = [{"params": decay, "weight_decay": cfg.adam_weight_decay},
              {"params": split.adam_no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.adam_lr, betas=cfg.adam_betas, eps=cfg.adam_eps)
