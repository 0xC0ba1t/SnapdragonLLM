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
