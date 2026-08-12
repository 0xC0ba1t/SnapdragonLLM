"""Console and JSONL metric loggers."""
from __future__ import annotations

import json
import math
from pathlib import Path

from snapdragon.train.callbacks.base import Callback
from snapdragon.utils.logging import get_logger

log = get_logger("train")


def format_step(metrics: dict, total_steps: int) -> str:
    parts = [f"step {metrics['step']:>6d}/{total_steps}", f"loss {metrics['loss']:.4f}"]
    if "lr" in metrics:
        parts.append(f"lr {metrics['lr']:.2e}")
    if "grad_norm" in metrics:
        parts.append(f"gnorm {metrics['grad_norm']:.2f}")
    if "tokens_per_sec" in metrics:
        parts.append(f"{metrics['tokens_per_sec'] / 1e3:.1f}k tok/s")
    if "mfu" in metrics:
        parts.append(f"mfu {metrics['mfu']:.1%}")
    if "mem_peak_gb" in metrics:
        parts.append(f"mem {metrics['mem_peak_gb']:.1f}GB")
    if metrics.get("skipped"):
        parts.append("SKIPPED")
    return " | ".join(parts)


class ConsoleLogger(Callback):
    priority = 50

    def __init__(self, every: int = 10) -> None:
        self.every = every

    def on_step_end(self, trainer, metrics: dict) -> None:
        if trainer.is_main and (metrics["step"] % self.every == 0 or metrics["step"] == trainer.total_steps):
            log.info(format_step(metrics, trainer.total_steps))


def _jsonable(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None  # keep the file valid JSON even when a loss blew up
    return v


class JsonlLogger(Callback):
    """Appends one JSON object per logged step to ``<run_dir>/metrics.jsonl``."""

    priority = 55

    def __init__(self, run_dir: str | Path, every: int = 1) -> None:
        self.path = Path(run_dir) / "metrics.jsonl"
        self.every = every

    def on_train_start(self, trainer) -> None:
        if trainer.is_main:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer, metrics: dict) -> None:
        if trainer.is_main and metrics["step"] % self.every == 0:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({k: _jsonable(v) for k, v in metrics.items()}) + "\n")
