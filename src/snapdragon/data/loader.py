"""Training loader: sequential batches with a checkpointable position."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from snapdragon.data.batches import BatchBuilder


@dataclass
class Batch:
    x: torch.Tensor  # (B, T) input tokens
    y: torch.Tensor  # (B, T) next-token targets


class TrainLoader:
    """Iterates batch 0, 1, 2, ... and can be resumed at any batch number via state_dict."""

    def __init__(self, builder: BatchBuilder, start_batch: int = 0) -> None:
        self.builder = builder
        self.batch_index = start_batch  # number of the next batch to hand out

    def __iter__(self) -> TrainLoader:
        return self

    def __next__(self) -> Batch:
        x, y = self.builder.get_batch(self.batch_index)
        self.batch_index += 1
        return Batch(x, y)

    def state_dict(self) -> dict:
        return {"batch_index": self.batch_index}

    def load_state_dict(self, state: dict) -> None:
        self.batch_index = int(state["batch_index"])
