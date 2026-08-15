"""Callbacks that watch training health: GPU memory, weight norm, divergence."""
from __future__ import annotations

import torch

from snapdragon.train.callbacks.base import Callback


class MemoryCallback(Callback):
    """Adds the CUDA peak allocated memory (GB) since the previous step. No-op on CPU."""

    priority = 10

    def on_train_start(self, trainer) -> None:
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def on_step_end(self, trainer, metrics: dict) -> None:
        if trainer.device.type == "cuda":
            metrics["mem_peak_gb"] = torch.cuda.max_memory_allocated(trainer.device) / 1e9
            metrics["mem_reserved_gb"] = torch.cuda.memory_reserved(trainer.device) / 1e9


class WeightNormCallback(Callback):
    """Adds the global parameter L2 norm every ``every`` steps (slowly growing is healthy;
    a sudden jump is an early sign of instability)."""

    priority = 10

    def __init__(self, every: int = 100) -> None:
        self.every = every

    @torch.no_grad()
    def on_step_end(self, trainer, metrics: dict) -> None:
        if metrics["step"] % self.every:
            return
        params = [p.detach() for p in trainer.raw_model.parameters()]
        norms = torch._foreach_norm(params)
        metrics["param_norm"] = float(torch.linalg.vector_norm(torch.stack([n.float() for n in norms])))


class DivergenceError(RuntimeError):
    """Training stopped because too many consecutive steps were skipped."""


class DivergenceGuard(Callback):
    """Aborts when ``max_consecutive_skips`` steps in a row were skipped (NaN/inf or loss spikes).

    One bad batch is handled by skipping it; a run of them means the weights themselves are bad, and
    continuing only burns compute. The exception propagates so the run fails loudly.
    """

    priority = 20

    def __init__(self, max_consecutive_skips: int = 10) -> None:
        self.max = max_consecutive_skips
        self.streak = 0

    def on_step_end(self, trainer, metrics: dict) -> None:
        self.streak = self.streak + 1 if metrics.get("skipped") else 0
        if self.streak >= self.max:
            raise DivergenceError(
                f"{self.streak} consecutive skipped steps at step {metrics['step']} "
                f"(last loss {metrics['loss']}); the run has diverged"
            )
