#!/usr/bin/env python3
"""Print the predicted GPU memory for a training config (no GPU, no model allocation).

    python scripts/estimate_memory.py --config configs/train/pretrain-16gb.yaml
    python scripts/estimate_memory.py --config configs/train/pretrain-16gb.yaml train.micro_batch_size=2
"""
from __future__ import annotations

import argparse

from snapdragon.config.io import load_config
from snapdragon.model.flops import flops_per_token
from snapdragon.train.memory import estimate_memory


def main(argv: list[str] | None = None) -> float:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget-gb", type=float, default=None, help="exit non-zero if the estimate exceeds this")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    est = estimate_memory(cfg)
    m, t = cfg.model, cfg.train
    print(f"{t.run_name}: {m.num_params() / 1e6:.0f}M params, micro-batch {t.micro_batch_size} x {t.seq_len}, "
          f"act_ckpt={m.act_ckpt}, 8-bit optimizer={cfg.optim.eight_bit}, offload={cfg.optim.offload_states}")
    for name, gb in est.rows():
        print(f"  {name:<28}{gb:>7.2f} GB")
    recompute = {"none": "none", "sac": "none", "selective": "selective", "full": "full"}[m.act_ckpt]
    flops = flops_per_token(m, t.seq_len, recompute)
    print(f"  compute: {flops / 1e9:.2f} GFLOPs/token -> {t.max_tokens * flops:.2e} FLOPs for {t.max_tokens:.1e} tokens")
    if args.budget_gb is not None and est.total > args.budget_gb:
        raise SystemExit(f"estimate {est.total:.1f} GB exceeds the {args.budget_gb} GB budget")
    return est.total


if __name__ == "__main__":
    main()
