"""Training loader: sequential batches with a checkpointable position."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from snapdragon.data.batches import BatchBuilder
from snapdragon.data.sampler import batch_index_for, global_samples


@dataclass
class Batch:
    x: torch.Tensor  # (B, T) input tokens
    y: torch.Tensor  # (B, T) next-token targets
    position_ids: torch.Tensor | None = None  # (B, T) optional, position within each document

    def regroup(self, seq_len: int) -> Batch:
        """Cut each (B, T) row into T // seq_len shorter rows: same tokens, shorter contexts."""
        B, T = self.x.shape
        if seq_len == T:
            return self
        if T % seq_len:
            raise ValueError(f"seq_len {seq_len} must divide the batch length {T}")

        def cut(t):
            return None if t is None else t.reshape(B * (T // seq_len), seq_len)

        return Batch(cut(self.x), cut(self.y), cut(self.position_ids))

    def with_doc_info(self, eos_id: int) -> Batch:
        """Attach per-document position ids (restarting after each <|eos|>) for packed training."""
        from snapdragon.data.packing import position_ids_from_tokens

        return Batch(self.x, self.y, position_ids_from_tokens(self.x, eos_id))

    def pin(self) -> Batch:
        """Page-lock host memory so the H2D copy can be asynchronous."""
        pos = self.position_ids.pin_memory() if self.position_ids is not None else None
        return Batch(self.x.pin_memory(), self.y.pin_memory(), pos)

    def to(self, device: torch.device | str, non_blocking: bool = True) -> Batch:
        pos = self.position_ids.to(device, non_blocking=non_blocking) if self.position_ids is not None else None
        return Batch(
            self.x.to(device, non_blocking=non_blocking), self.y.to(device, non_blocking=non_blocking), pos
        )


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
        b = self.builder
        return {"batch_index": self.batch_index,
                "global_samples": global_samples(self.batch_index, b.micro_batch_size, b.world_size)}

    def load_state_dict(self, state: dict) -> None:
        b = self.builder
        if "global_samples" in state:  # layout-independent position (see PrefetchLoader.load_state_dict)
            self.batch_index = batch_index_for(state["global_samples"], b.micro_batch_size, b.world_size)
        else:
            self.batch_index = int(state["batch_index"])
