#!/usr/bin/env python3
"""Benchmark training throughput for a config on this machine (synthetic data).

    python scripts/benchmark.py --config configs/train/pretrain-16gb.yaml --steps 3
    python scripts/benchmark.py --config configs/train/pretrain-16gb.yaml train.micro_batch_size=2 model.act_ckpt=full
"""
from __future__ import annotations

import argparse

from snapdragon.config.io import load_config
from snapdragon.train.benchmark import BenchmarkResult, run_benchmark
from snapdragon.train.memory import estimate_memory
from snapdragon.utils.device import configure_backends


def main(argv: list[str] | None = None) -> BenchmarkResult:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    configure_backends()
    cfg = load_config(args.config, args.overrides)
    est = estimate_memory(cfg)
    print(f"{cfg.train.run_name}: predicted peak {est.total:.1f} GB, "
          f"{cfg.train.grad_accum_steps()} micro-batches of {cfg.train.micro_batch_size} x {cfg.train.seq_len} per step")
    result = run_benchmark(cfg, args.steps, args.warmup, args.device, False if args.no_compile else None)
    print(result.summary())
    days = cfg.train.max_tokens / result.tokens_per_sec / 86400
    print(f"=> {cfg.train.max_tokens:.1e} tokens would take about {days:.1f} days at this speed")
    return result


if __name__ == "__main__":
    main()
