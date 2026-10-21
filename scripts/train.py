#!/usr/bin/env python3
"""Train a model from a YAML config.

    python scripts/train.py --config configs/train/pretrain-16gb.yaml
    python scripts/train.py --config configs/train/debug-cpu.yaml train.max_tokens=100000
    torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/train/pretrain-16gb.yaml

Trailing ``section.key=value`` arguments override the YAML. Re-running the same command resumes
from the newest checkpoint of the run (``train.resume=none`` disables that).
"""
from __future__ import annotations

import argparse

from snapdragon.config.io import load_config
from snapdragon.distributed.env import cleanup_distributed
from snapdragon.train.setup import build_trainer, device_summary
from snapdragon.utils.device import configure_backends
from snapdragon.utils.logging import get_logger, setup_logging

log = get_logger("train")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:1 (default: auto)")
    ap.add_argument("overrides", nargs="*", help="section.key=value")
    args = ap.parse_args(argv)

    configure_backends()  # before any CUDA allocation so the allocator settings take effect
    setup_logging()
    cfg = load_config(args.config, args.overrides)
    trainer = build_trainer(cfg, device=args.device)
    if trainer.is_main:
        log.info("device: %s | ranks: %d | steps: %d | grad accum: %d", device_summary(trainer.device),
                 trainer.world_size, trainer.total_steps, trainer.grad_accum_steps)
    try:
        trainer.fit()
    finally:
        close = getattr(trainer.train_loader, "close", None)
        if close:
            close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
