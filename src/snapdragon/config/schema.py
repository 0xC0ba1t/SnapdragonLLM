"""Top-level config tying the sections together."""
from __future__ import annotations

from dataclasses import dataclass, field

from snapdragon.config.data import DataConfig
from snapdragon.config.model import ModelConfig
from snapdragon.config.optim import OptimConfig
from snapdragon.config.train import TrainConfig


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Cross-section consistency checks (single-section checks live in each dataclass)."""
        if self.train.seq_len > self.model.max_seq_len:
            raise ValueError(
                f"train.seq_len {self.train.seq_len} exceeds model.max_seq_len {self.model.max_seq_len}"
            )
        if self.data.seq_len != self.train.seq_len:
            raise ValueError(
                f"data.seq_len {self.data.seq_len} must equal train.seq_len {self.train.seq_len}"
            )
        if self.data.doc_masking and self.model.attn_impl != "flex":
            raise ValueError("data.doc_masking requires model.attn_impl='flex'")
