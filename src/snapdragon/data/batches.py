"""Builds (x, y) training batches from datasets via the sampler. No threads, fully deterministic."""
from __future__ import annotations

import numpy as np
import torch

from snapdragon.data.memmap_dataset import TokenDataset
from snapdragon.data.sampler import Sampler, batch_sample_indices


class BatchBuilder:
    """Batch ``n`` is a pure function of (datasets, weights, seed, seq_len, B, rank, world, n)."""

    def __init__(
        self,
        datasets: list[TokenDataset],
        weights: list[float],
        seq_len: int,
        micro_batch_size: int,
        seed: int = 1337,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.datasets = datasets
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.rank, self.world_size = rank, world_size
        self.sampler = Sampler([d.num_windows(seq_len) for d in datasets], weights, seed)

    def get_batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        for g in batch_sample_indices(n, self.micro_batch_size, self.rank, self.world_size):
            src, window = self.sampler.locate(g)
            rows.append(self.datasets[src].get_window(window, self.seq_len))
        w = torch.from_numpy(np.stack(rows))
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()
