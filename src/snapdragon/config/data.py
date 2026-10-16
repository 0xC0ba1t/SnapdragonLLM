"""Data pipeline configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceConfig:
    """One tokenized dataset (a directory of .bin shards plus an index) in the mixture."""

    name: str
    path: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"source {self.name!r}: weight must be positive")


@dataclass
class DataConfig:
    sources: list[SourceConfig] = field(default_factory=list)
    sft_path: str = ""  # jsonl of {"messages": [...]} conversations, used when train.task == "sft"
    tokenizer_path: str = "data/tokenizer/snapdragon-32k.json"
    seq_len: int = 2048
    doc_masking: bool = False  # attend only within a document (needs the flex attention path)
    prefetch_batches: int = 4
    pin_memory: bool = True
    val_tokens: int = 10_000_000
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate source names: {names}")

    def normalized_weights(self) -> dict[str, float]:
        total = sum(s.weight for s in self.sources)
        return {s.name: s.weight / total for s in self.sources}
