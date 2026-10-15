"""In-memory loader over packed SFT rows: deterministic, reshuffled every epoch, resumable."""
from __future__ import annotations

import torch

from snapdragon.data.loader import Batch
from snapdragon.data.sampler import EpochPermutation, batch_sample_indices
from snapdragon.sft.packing import PackedRows


class SFTLoader:
    """Same contract as the pretraining loaders (``next()``, ``state_dict``, ``load_state_dict``) so the
    Trainer needs no SFT-specific code. Row order is a pure function of (seed, global sample index)."""

    def __init__(self, rows: PackedRows, micro_batch_size: int, seed: int = 1337,
                 rank: int = 0, world_size: int = 1, start_batch: int = 0) -> None:
        if rows.x.shape[0] == 0:
            raise ValueError("no SFT rows: the dataset is empty after filtering")
        self.rows, self.micro_batch_size = rows, micro_batch_size
        self.rank, self.world_size = rank, world_size
        self.perm = EpochPermutation(rows.x.shape[0], seed, stream=0)
        self.batch_index = start_batch

    def __iter__(self) -> SFTLoader:
        return self

    def __next__(self) -> Batch:
        idx = [self.perm.index(g) for g in
               batch_sample_indices(self.batch_index, self.micro_batch_size, self.rank, self.world_size)]
        self.batch_index += 1
        sel = torch.tensor(idx)
        return Batch(self.rows.x[sel], self.rows.y[sel], self.rows.position_ids[sel])

    def state_dict(self) -> dict:
        return {"batch_index": self.batch_index}

    def load_state_dict(self, state: dict) -> None:
        self.batch_index = int(state["batch_index"])
