"""Optimizer and LR schedule configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

SCHEDULE_KINDS = ("wsd", "cosine", "linear")


@dataclass
class SchedulerConfig:
    kind: str = "wsd"  # warmup-stable-decay: can stop or extend anywhere, then run a short decay
    warmup_steps: int = 500
    decay_fraction: float = 0.2  # fraction of total steps spent decaying (wsd only)
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_KINDS:
            raise ValueError(f"scheduler kind must be one of {SCHEDULE_KINDS}, got {self.kind!r}")
        if not 0.0 <= self.decay_fraction <= 1.0:
            raise ValueError("decay_fraction must be in [0, 1]")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")


@dataclass
class OptimConfig:
    # Muon drives the hidden 2-D matrices, AdamW (optionally 8-bit) the embeddings and gains.
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    adam_lr: float = 6e-4
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.1
    eight_bit: bool = True
    offload_states: bool = False  # keep optimizer state on CPU (12GB cards)
    grad_clip: float = 1.0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self) -> None:
        if self.muon_lr <= 0 or self.adam_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be >= 0 (0 disables clipping)")
