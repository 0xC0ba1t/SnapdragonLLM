"""Sequence-length warmup: start with short contexts and grow to the full length.

Short sequences make early, high-gradient-noise steps cheaper in attention and noticeably more
stable at large learning rates (Li et al., "The Stability-Efficiency Dilemma", 2022). To keep the
token count per step constant the loader still reads full-length windows; the trainer cuts each into
``max_len // seq_len`` shorter rows.
"""
from __future__ import annotations

from snapdragon.config.train import TrainConfig


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def seq_len_at(tokens_seen: int, cfg: TrainConfig) -> int:
    """Sequence length to train at after ``tokens_seen`` tokens (always a divisor of cfg.seq_len)."""
    if not cfg.seq_len_warmup_tokens or tokens_seen >= cfg.seq_len_warmup_tokens:
        return cfg.seq_len
    frac = tokens_seen / cfg.seq_len_warmup_tokens
    target = cfg.seq_len_start + (cfg.seq_len - cfg.seq_len_start) * frac
    allowed = [d for d in divisors(cfg.seq_len) if d >= cfg.seq_len_start]
    below = [d for d in allowed if d <= target]
    return below[-1] if below else allowed[0]
