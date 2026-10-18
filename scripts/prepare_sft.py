#!/usr/bin/env python3
"""Normalise an instruction dataset into chat jsonl for SFT.

    python scripts/prepare_sft.py --hf yahma/alpaca-cleaned --format alpaca --out data/sft/alpaca.jsonl
    python scripts/prepare_sft.py --files mine.jsonl --format sharegpt --out data/sft/mine.jsonl --shuffle-seed 1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from snapdragon.sft.formats import normalise


def read_rows(args) -> list[dict]:
    if args.hf:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("install the 'data' extra: pip install -e .[data]") from e
        name, _, config = args.hf.partition(":")
        return list(load_dataset(name, config or None, split=args.split))
    rows = []
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="dataset name[:config] on the HuggingFace hub")
    src.add_argument("--files", nargs="+", help="local jsonl files")
    ap.add_argument("--split", default="train")
    ap.add_argument("--format", required=True, choices=["alpaca", "sharegpt", "messages"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    args = ap.parse_args(argv)

    out = list(normalise(read_rows(args), args.format))
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} conversations to {args.out}")
    return len(out)


if __name__ == "__main__":
    main()
