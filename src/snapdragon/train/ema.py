"""Exponential moving average of the weights: usually evaluates (and generalises) better than the raw
weights mid-run, and smooths the noise between checkpoints. Costs one extra fp32 copy of the model."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn

from snapdragon.train.callbacks.base import Callback


class WeightEMA:
    """Shadow copy ``ema = decay * ema + (1 - decay) * weights``, updated with one fused foreach op.

    ``warmup`` ramps the decay from ~0 so the first steps are not dominated by the random init:
    ``decay_t = min(decay, (1 + t) / (10 + t))``.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: bool = True) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay, self.warmup, self.num_updates = decay, warmup, 0
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters()}

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        t = self.num_updates
        return min(self.decay, (1 + t) / (10 + t))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        params = [p.detach().float() for _, p in model.named_parameters()]
        torch._foreach_lerp_(list(self.shadow.values()), params, 1.0 - self.current_decay())
        self.num_updates += 1

    @contextlib.contextmanager
    def applied(self, model: nn.Module) -> Iterator[None]:
        """Temporarily load the EMA weights into ``model`` (for evaluation), then restore the originals."""
        backup = {n: p.detach().clone() for n, p in model.named_parameters()}
        try:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(self.shadow[n].to(p.dtype))
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(backup[n])

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.float() for k, v in state["shadow"].items()}
        self.num_updates = state["num_updates"]


class EMACallback(Callback):
    """Keeps a WeightEMA on the trainer (registered for checkpointing) and updates it every step."""

    priority = 15

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema: WeightEMA | None = None

    def on_train_start(self, trainer) -> None:
        if self.ema is None:
            self.ema = WeightEMA(trainer.raw_model, self.decay)
        trainer.extra_state["ema"] = self.ema

    def on_step_end(self, trainer, metrics: dict) -> None:
        if not metrics.get("skipped"):
            self.ema.update(trainer.raw_model)
