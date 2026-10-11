#!/usr/bin/env python3
"""Generate text from a checkpoint.

    python scripts/generate.py --checkpoint runs/x/checkpoints --tokenizer data/tokenizer/snapdragon-32k.json \\
        --prompt "The capital of France is" --max-new-tokens 60 --temperature 0.7 --top-p 0.9
"""
from __future__ import annotations

import argparse
import sys
import time

from snapdragon.checkpoint.model_io import load_model_from_checkpoint
from snapdragon.inference.sampling import SamplingConfig
from snapdragon.inference.text import stream_text
from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.utils.device import get_device


def main(argv: list[str] | None = None, out=None) -> str:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--stop", action="append", default=[], help="stop string (repeatable)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stats", action="store_true", help="print tokens/s to stderr")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device)
    tok = FastTokenizer.from_file(args.tokenizer)
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p, args.min_p, args.repetition_penalty)

    pieces: list[str] = []
    t0 = time.perf_counter()
    for chunk in stream_text(model, tok, args.prompt, args.max_new_tokens, sampling, args.stop, args.seed):
        pieces.append(chunk)
        out.write(chunk)
        out.flush()
    out.write("\n")
    if args.stats:
        n = len(tok.encode("".join(pieces)))
        print(f"{n} tokens in {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    return "".join(pieces)


if __name__ == "__main__":
    main()
