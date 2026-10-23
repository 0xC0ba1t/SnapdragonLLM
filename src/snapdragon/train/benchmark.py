"""Measure training throughput on synthetic data: tokens/s, step time, MFU and peak memory.

Synthetic batches keep the data pipeline out of the measurement, so this isolates model, optimizer and
memory-saving settings; use it to pick ``micro_batch_size`` / ``act_ckpt`` for a given GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from snapdragon.config.schema import Config
from snapdragon.data.loader import Batch
from snapdragon.model.compile import compile_blocks, should_compile
from snapdragon.model.factory import build_model
from snapdragon.model.flops import mfu, peak_tflops_for
from snapdragon.optim.factory import build_optimizer
from snapdragon.optim.schedulers import LRSchedule
from snapdragon.train.trainer import Trainer
from snapdragon.utils.device import get_device


class RandomLoader:
    """Endless random token batches (content is irrelevant for throughput)."""

    def __init__(self, batch_size: int, seq_len: int, vocab: int, seed: int = 0) -> None:
        self.shape, self.vocab = (batch_size, seq_len + 1), vocab
        self.gen = torch.Generator().manual_seed(seed)
        self.batch_index = 0

    def __next__(self) -> Batch:
        w = torch.randint(0, self.vocab, self.shape, generator=self.gen)
        self.batch_index += 1
        return Batch(w[:, :-1].contiguous(), w[:, 1:].contiguous())

    def state_dict(self) -> dict:
        return {"batch_index": self.batch_index}

    def load_state_dict(self, state: dict) -> None:
        self.batch_index = state["batch_index"]


@dataclass
class BenchmarkResult:
    steps: int
    seconds_per_step: float
    tokens_per_sec: float
    mfu: float | None
    peak_memory_gb: float | None
    tokens_per_step: int

    def summary(self) -> str:
        parts = [f"{self.tokens_per_sec / 1e3:.2f}k tok/s", f"{self.seconds_per_step:.2f} s/step"]
        if self.mfu is not None:
            parts.append(f"MFU {self.mfu:.1%}")
        if self.peak_memory_gb is not None:
            parts.append(f"peak {self.peak_memory_gb:.1f} GB")
        return " | ".join(parts)


def run_benchmark(cfg: Config, steps: int = 5, warmup: int = 2, device: str | None = None,
                  compile_model: bool | None = None) -> BenchmarkResult:
    """Run ``warmup + steps`` real optimizer steps (full grad accumulation) and time the last ``steps``."""
    dev = get_device(device)
    model = build_model(cfg).to(dev)
    if compile_model if compile_model is not None else should_compile(cfg.train.compile):
        compile_blocks(model)
    opt = build_optimizer(model, cfg.optim)
    sched = LRSchedule(opt, cfg.optim.scheduler, max(cfg.train.total_steps, warmup + steps + 1))
    loader = RandomLoader(cfg.train.micro_batch_size, cfg.train.seq_len, cfg.model.vocab_size)
    trainer = Trainer(cfg, model, opt, sched, loader, dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for _ in range(warmup):
        trainer.train_step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(steps):
        trainer.train_step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = (time.perf_counter() - t0) / steps
    tokens = cfg.train.global_batch_tokens
    tps = tokens / dt
    peak = torch.cuda.get_device_name(dev) if dev.type == "cuda" else None
    return BenchmarkResult(
        steps=steps,
        seconds_per_step=dt,
        tokens_per_sec=tps,
        mfu=mfu(tps, cfg.model, peak_tflops_for(peak), cfg.train.seq_len) if peak and peak_tflops_for(peak) else None,
        peak_memory_gb=torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None,
        tokens_per_step=tokens,
    )
