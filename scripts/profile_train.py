#!/usr/bin/env python3
"""Profile a few training steps and write a Chrome trace, a top-ops table and (CUDA) a memory snapshot.

    python scripts/profile_train.py --config configs/train/pretrain-16gb.yaml --out profiles/16gb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from snapdragon.config.io import load_config
from snapdragon.model.factory import build_model
from snapdragon.optim.factory import build_optimizer
from snapdragon.optim.schedulers import LRSchedule
from snapdragon.train.benchmark import RandomLoader
from snapdragon.train.trainer import Trainer
from snapdragon.utils.device import configure_backends, get_device
from snapdragon.utils.profiling import memory_history, profile_run


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="profiles/run")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=2, help="profiled optimizer steps (after 1 wait + 1 warmup)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    configure_backends()
    cfg = load_config(args.config, args.overrides)
    dev = get_device(args.device)
    model = build_model(cfg).to(dev)
    opt = build_optimizer(model, cfg.optim)
    sched = LRSchedule(opt, cfg.optim.scheduler, 1000)
    loader = RandomLoader(cfg.train.micro_batch_size, cfg.train.seq_len, cfg.model.vocab_size)
    trainer = Trainer(cfg, model, opt, sched, loader, dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with memory_history(out / "memory_snapshot.pickle"), profile_run(out, wait=1, warmup=1, active=args.steps) as prof:
        for _ in range(2 + args.steps):
            trainer.train_step()
            prof.step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    print(f"wrote {out / 'trace.json'} and {out / 'top_ops.txt'}")
    return out


if __name__ == "__main__":
    main()
