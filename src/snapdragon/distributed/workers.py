"""Worker functions used by the multi-process tests (see testing.py for why they live here)."""
from __future__ import annotations

import torch

from snapdragon.config.model import ModelConfig
from snapdragon.distributed.env import DistEnv
from snapdragon.distributed.wrap import wrap_ddp
from snapdragon.model.factory import build_model

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def _data(n_rows: int):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 64, (n_rows, 16), generator=g)
    return x, torch.roll(x, -1, dims=1)


def ddp_grads(rank: int, world_size: int) -> dict:
    """Each rank backprops its own slice of an 8-row batch through a DDP-wrapped model."""
    model = wrap_ddp(build_model(CFG), DistEnv(rank, rank, world_size))
    x, y = _data(8)
    per = 8 // world_size
    xs, ys = x[rank * per : (rank + 1) * per], y[rank * per : (rank + 1) * per]
    loss, _, _ = model(xs, ys)
    loss.backward()
    return {n: p.grad.clone() for n, p in model.module.named_parameters()}


def single_process_grads() -> dict:
    model = build_model(CFG)
    x, y = _data(8)
    loss, _, _ = model(x, y)
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()}
