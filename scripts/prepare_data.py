#!/usr/bin/env python3
"""Download, filter, de-duplicate and tokenize a dataset into uint16 shards.

    python scripts/prepare_data.py --spec hf:HuggingFaceFW/fineweb-edu:sample-10BT \
        --tokenizer data/tokenizer/snapdragon-32k.json --out data/fineweb-edu --max-tokens 10_000_000_000
"""
from __future__ import annotations

import argparse
from collections import Counter

from snapdragon.data.dedup import ExactDedup
from snapdragon.data.download import iter_documents
from snapdragon.data.filters import QualityFilter
from snapdragon.data.manifest import DatasetManifest
from snapdragon.data.tokenize_shards import tokenize_to_shards
from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.utils.logging import get_logger, setup_logging

log = get_logger("prepare_data")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="hf:<name>[:<config>] or files:<a>,<b>")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=lambda s: int(s.replace("_", "")), default=None)
    ap.add_argument("--shard-tokens", type=lambda s: int(s.replace("_", "")), default=100_000_000)
    ap.add_argument("--val-tokens", type=lambda s: int(s.replace("_", "")), default=10_000_000)
    ap.add_argument("--no-filter", action="store_true", help="skip quality heuristics")
    ap.add_argument("--no-dedup", action="store_true", help="skip exact de-duplication")
    ap.add_argument("--verify", action="store_true", help="only verify an existing output directory")
    args = ap.parse_args(argv)
    setup_logging()

    if args.verify:
        m = DatasetManifest.load(args.out)
        m.verify(args.out)
        log.info("ok: %d shards, %.2fB tokens", len(m.shards), m.total_tokens() / 1e9)
        return

    tok = FastTokenizer.from_file(args.tokenizer)
    stats: Counter = Counter()
    m = tokenize_to_shards(
        iter_documents(args.spec),
        tok,
        args.out,
        shard_tokens=args.shard_tokens,
        val_tokens=args.val_tokens,
        max_tokens=args.max_tokens,
        quality_filter=None if args.no_filter else QualityFilter(),
        dedup=None if args.no_dedup else ExactDedup(),
        stats=stats,
    )
    m.verify(args.out)
    log.info("document stats: %s", dict(stats))
    log.info("train tokens: %d, val tokens: %d", m.total_tokens("train"), m.total_tokens("val"))


if __name__ == "__main__":
    main()
