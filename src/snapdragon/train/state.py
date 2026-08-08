"""Mutable training progress that must survive a restart."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainState:
    step: int = 0  # optimizer steps completed
    tokens_seen: int = 0  # global training tokens consumed (all ranks)
    skipped_steps: int = 0  # steps dropped for non-finite grads or loss spikes
    loss_ema: float = 0.0  # smoothed training loss, used for spike detection
    best_val_loss: float = float("inf")

    def update_loss_ema(self, loss: float, beta: float = 0.98) -> float:
        """Bias-aware EMA: the first value initialises it instead of decaying from zero."""
        self.loss_ema = loss if self.loss_ema == 0.0 else beta * self.loss_ema + (1 - beta) * loss
        return self.loss_ema

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            if not hasattr(self, key):
                raise KeyError(f"unknown train state field {key!r}")
            setattr(self, key, value)
