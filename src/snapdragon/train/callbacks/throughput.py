"""Throughput (tokens/s), step time and MFU."""
from __future__ import annotations

import time

import torch

from snapdragon.model.flops import mfu, peak_tflops_for
from snapdragon.train.callbacks.base import Callback


class ThroughputCallback(Callback):
    """Adds ``step_time``, ``tokens_per_sec`` and (if the GPU's peak is known) ``mfu`` to the metrics.

    The first step is excluded from timing: it includes allocator warm-up and compilation.
    """

    priority = 10

    def __init__(self, peak_tflops: float | None = None) -> None:
        self.peak_tflops = peak_tflops
        self._last_time: float | None = None
        self._last_tokens = 0

    def on_train_start(self, trainer) -> None:
        if self.peak_tflops is None and trainer.device.type == "cuda":
            self.peak_tflops = peak_tflops_for(torch.cuda.get_device_name(trainer.device))
        self._last_time = None
        self._last_tokens = trainer.state.tokens_seen

    def on_step_end(self, trainer, metrics: dict) -> None:
        now = time.perf_counter()
        tokens = trainer.state.tokens_seen
        if self._last_time is not None:
            dt = now - self._last_time
            tps = (tokens - self._last_tokens) / dt if dt > 0 else 0.0
            metrics["step_time"] = dt
            metrics["tokens_per_sec"] = tps
            if self.peak_tflops:
                # tokens/s is global; MFU is per device, so divide by the number of ranks
                metrics["mfu"] = mfu(tps / trainer.world_size, trainer.raw_model.cfg, self.peak_tflops,
                                     trainer.cfg.train.seq_len)
        self._last_time, self._last_tokens = now, tokens
