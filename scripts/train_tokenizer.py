#!/usr/bin/env python3
"""Train the 32k byte-level BPE tokenizer.

    python scripts/train_tokenizer.py --hf HuggingFaceFW/fineweb-edu:sample-10BT --max-gb 2
    python scripts/train_tokenizer.py --files corpus.jsonl --vocab-size 32768
"""
from __future__ import annotations

import argparse
import time

from snapdragon.tokenizer.corpus import iter_hf_texts, iter_local_texts, take_bytes
from snapdragon.tokenizer.fast import FastTokenizer, train_fast_tokenizer
from snapdragon.utils.logging import get_logger, setup_logging

log = get_logger("train_tokenizer")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="dataset as name[:config], e.g. HuggingFaceFW/fineweb-edu:sample-10BT")
    src.add_argument("--files", nargs="+", help="local .txt / .jsonl files")
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--max-gb", type=float, default=2.0, help="stop reading after about this many GB of text")
    ap.add_argument("--out", default="data/tokenizer/snapdragon-32k.json")
    args = ap.parse_args(argv)

    setup_logging()
    if args.hf:
        name, _, config = args.hf.partition(":")
        texts = iter_hf_texts(name, config or None)
    else:
        texts = iter_local_texts(args.files)
    texts = take_bytes(texts, int(args.max_gb * 1e9))

    t0 = time.time()
    tok = FastTokenizer(train_fast_tokenizer(texts, args.vocab_size))
    tok.save(args.out)
    log.info("trained %d-token tokenizer in %.0fs -> %s", tok.vocab_size, time.time() - t0, args.out)


if __name__ == "__main__":
    main()
