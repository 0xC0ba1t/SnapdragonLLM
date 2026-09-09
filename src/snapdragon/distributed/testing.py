"""Helpers for running multi-process (gloo, CPU) distributed tests with torch.multiprocessing.

Workers live in the library, not in test files, because spawned children must be able to import them.
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def gloo_available() -> bool:
    """The Windows PyTorch wheels ship without a usable gloo transport; tests skip there (Linux CI runs them)."""
    return sys.platform != "win32" and dist.is_available() and dist.is_gloo_available()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank: int, world_size: int, port: int, fn: Callable, out_dir: str, args: tuple) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        result = fn(rank, world_size, *args)
        torch.save(result, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def run_distributed(fn: Callable, world_size: int, *args) -> list:
    """Run ``fn(rank, world_size, *args)`` in ``world_size`` processes; return each rank's result."""
    port = free_port()
    with tempfile.TemporaryDirectory() as out_dir:
        mp.spawn(_entry, args=(world_size, port, fn, out_dir, args), nprocs=world_size, join=True)
        return [torch.load(os.path.join(out_dir, f"rank{r}.pt"), weights_only=False) for r in range(world_size)]
