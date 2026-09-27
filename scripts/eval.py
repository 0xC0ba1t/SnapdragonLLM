#!/usr/bin/env python3
"""Evaluate a trained checkpoint on benchmarks and/or validation perplexity.

    python scripts/eval.py --checkpoint runs/snapdragon-500m-16gb/checkpoints --tokenizer data/tokenizer/snapdragon-32k.json \\
        --tasks hellaswag arc_easy arc_challenge piqa lambada --out results.json
    python scripts/eval.py --checkpoint runs/x/checkpoints --perplexity data/fineweb-edu
"""
from __future__ import annotations

import argparse
import json

from snapdragon.checkpoint.model_io import load_model_from_checkpoint
from snapdragon.data.memmap_dataset import TokenDataset
from snapdragon.eval.perplexity import evaluate_perplexity
from snapdragon.eval.runner import ALL_TASKS, run_tasks
from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.train.precision import autocast_context
from snapdragon.utils.device import get_device
from snapdragon.utils.logging import get_logger, setup_logging

log = get_logger("eval")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a step directory or a run's checkpoints directory")
    ap.add_argument("--tokenizer", help="tokenizer json (needed for --tasks)")
    ap.add_argument("--tasks", nargs="*", default=[], help=f"any of: {' '.join(ALL_TASKS)}")
    ap.add_argument("--perplexity", metavar="DATA_DIR", help="prepared dataset directory; evaluates its val split")
    ap.add_argument("--limit", type=int, default=0, help="examples per task (0 = all)")
    ap.add_argument("--max-batches", type=int, default=50, help="validation batches for --perplexity")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", help="write the results as json")
    args = ap.parse_args(argv)
    if args.tasks and not args.tokenizer:
        ap.error("--tasks needs --tokenizer")

    setup_logging()
    device = get_device(args.device)
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)
    log.info("loaded %.0fM-parameter model on %s", cfg.model.num_params() / 1e6, device)
    autocast = autocast_context(args.precision, device)
    results: dict = {}

    if args.tasks:
        tok = FastTokenizer.from_file(args.tokenizer)
        results.update(run_tasks(args.tasks, model, tok, device, args.limit, cfg.model.max_seq_len, autocast))
    if args.perplexity:
        ds = TokenDataset(args.perplexity, "val")
        r = evaluate_perplexity(model, ds, cfg.train.seq_len, 4, args.max_batches, device, autocast)
        results["perplexity"] = {"loss": r.loss, "ppl": r.ppl, "bits_per_token": r.bits_per_token, "tokens": r.tokens}

    for name, res in results.items():
        log.info("%s: %s", name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    main()
