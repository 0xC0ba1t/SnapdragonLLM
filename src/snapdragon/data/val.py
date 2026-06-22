"""Validation loader: fixed, unshuffled windows so val loss is comparable across runs."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from snapdragon.data.loader import Batch
from snapdragon.data.memmap_dataset import TokenDataset
from snapdragon.data.sampler import batch_sample_indices


class ValLoader:
    def __init__(
        self,
        dataset: TokenDataset,
        seq_len: int,
        micro_batch_size: int,
        max_batches: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.rank, self.world_size = rank, world_size
        per_round = micro_batch_size * world_size
        available = dataset.num_windows(seq_len) // per_round
        if available == 0:
            raise ValueError("validation split is too small for one batch per rank")
        self.num_batches = available if max_batches is None else min(available, max_batches)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[Batch]:
        for n in range(self.num_batches):
            rows = [
                self.dataset.get_window(i, self.seq_len)
                for i in batch_sample_indices(n, self.micro_batch_size, self.rank, self.world_size)
            ]
            w = torch.from_numpy(np.stack(rows))
            yield Batch(w[:, :-1].contiguous(), w[:, 1:].contiguous())
