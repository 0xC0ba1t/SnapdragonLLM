"""Learning-rate multipliers: warmup then WSD / cosine / linear decay."""
from __future__ import annotations

import math

from snapdragon.config.optim import SchedulerConfig


def lr_multiplier(cfg: SchedulerConfig, step: int, total_steps: int) -> float:
    """Factor in [min_lr_ratio, 1] applied to every group's base LR at optimizer step ``step``."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if step >= total_steps:
        return cfg.min_lr_ratio
    lo = cfg.min_lr_ratio
    if cfg.kind == "wsd":
        decay_steps = max(int(total_steps * cfg.decay_fraction), 1)
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / decay_steps
        return lo + (1.0 - lo) * (1.0 - math.sqrt(progress))  # fast initial drop, gentle tail
    span = max(total_steps - cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / span
    if cfg.kind == "cosine":
        return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lo + (1.0 - lo) * (1.0 - progress)  # linear


class LRSchedule:
    """Applies ``lr_multiplier`` to every param group of an optimizer.

    Each group's starting LR is remembered as ``base_lr``, so groups with different LRs
    (Muon vs AdamW) keep their ratio. The schedule is a pure function of the step; the only
    state to checkpoint is the step counter.
    """

    def __init__(self, optimizer, cfg: SchedulerConfig, total_steps: int, step: int = 0) -> None:
        self.optimizer, self.cfg, self.total_steps = optimizer, cfg, total_steps
        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])
        self.step_num = step
        self.apply(step)

    def apply(self, step: int) -> float:
        mult = lr_multiplier(self.cfg, step, self.total_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * mult
        self.step_num = step
        return mult

    def step(self) -> float:
        """Advance to the next optimizer step's LR."""
        return self.apply(self.step_num + 1)

    def state_dict(self) -> dict:
        return {"step_num": self.step_num, "total_steps": self.total_steps}

    def load_state_dict(self, state: dict) -> None:
        self.total_steps = state["total_steps"]
        self.apply(state["step_num"])
