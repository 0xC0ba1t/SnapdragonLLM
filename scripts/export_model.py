#!/usr/bin/env python3
"""Export a training checkpoint for release.

    python scripts/export_model.py --checkpoint runs/x/checkpoints --out release/snapdragon-500m \\
        --tokenizer data/tokenizer/snapdragon-32k.json --format safetensors hf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from snapdragon.checkpoint.model_io import load_model_from_checkpoint
from snapdragon.export.hf import export_hf
from snapdragon.export.safetensors_io import save_pretrained
from snapdragon.tokenizer.fast import FastTokenizer

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--format", nargs="+", choices=["safetensors", "hf"], default=["safetensors"])
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    args = ap.parse_args(argv)

    model, _ = load_model_from_checkpoint(args.checkpoint)
    dtype = DTYPES[args.dtype]
    written = []
    if "safetensors" in args.format:
        written.append(save_pretrained(model, Path(args.out) / "snapdragon", args.tokenizer, dtype))
    if "hf" in args.format:
        bos = eos = None
        if args.tokenizer:
            tok = FastTokenizer.from_file(args.tokenizer)
            bos, eos = tok.bos_id, tok.eos_id
        written.append(export_hf(model, Path(args.out) / "hf", args.tokenizer, bos, eos, dtype))
    for p in written:
        print(f"wrote {p}")
    return written


if __name__ == "__main__":
    main()
