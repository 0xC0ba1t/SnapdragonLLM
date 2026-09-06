"""Process-group setup driven by torchrun environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistEnv:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def detect_env(environ: dict | None = None) -> DistEnv:
    """Read RANK / LOCAL_RANK / WORLD_SIZE (set by torchrun); defaults describe a single process."""
    e = os.environ if environ is None else environ
    world = int(e.get("WORLD_SIZE", "1"))
    rank = int(e.get("RANK", "0"))
    local = int(e.get("LOCAL_RANK", "0"))
    if not 0 <= rank < world:
        raise ValueError(f"RANK {rank} is outside WORLD_SIZE {world}")
    return DistEnv(rank=rank, local_rank=local, world_size=world)


def init_distributed(env: DistEnv | None = None, backend: str | None = None) -> DistEnv:
    """Initialise the default process group (nccl on GPUs, gloo otherwise) and pin the CUDA device."""
    env = env or detect_env()
    if not env.distributed:
        return env
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(env.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return env


def device_for(env: DistEnv) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", env.local_rank)
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
