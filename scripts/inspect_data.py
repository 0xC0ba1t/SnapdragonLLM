#!/usr/bin/env python3
"""Summarise a prepared dataset directory: shards, tokens, documents, document-length distribution.

    python scripts/inspect_data.py data/fineweb-edu
    python scripts/inspect_data.py data/fineweb-edu --tokenizer data/tokenizer/snapdragon-32k.json --sample 3
"""
from __future__ import annotations

import argparse

import numpy as np

from snapdragon.data.manifest import DatasetManifest
from snapdragon.data.shards import read_index, read_tokens
from snapdragon.tokenizer.fast import FastTokenizer


def doc_lengths(directory, manifest: DatasetManifest, split: str) -> np.ndarray:
    """Token count of every document (including its <|eos|>) in a split."""
    lengths = [np.diff(read_index(f"{directory}/{s.name}")) for s in manifest.shards_for(split)]
    return np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--tokenizer", help="decode sample documents with this tokenizer")
    ap.add_argument("--sample", type=int, default=0, help="print this many decoded training documents")
    args = ap.parse_args(argv)

    m = DatasetManifest.load(args.directory)
    m.verify(args.directory)
    lines = [f"{args.directory}: vocab {m.vocab_size}, eos id {m.eos_id}, {len(m.shards)} shards, "
             f"{m.total_tokens() / 1e9:.3f}B tokens ({m.total_tokens() * 2 / 1e9:.2f} GB on disk)"]
    for split in ("train", "val"):
        n = doc_lengths(args.directory, m, split)
        if n.size == 0:
            continue
        p = np.percentile(n, [50, 90, 99])
        lines.append(f"  {split}: {m.total_tokens(split):,} tokens in {n.size:,} docs | tokens/doc mean {n.mean():.0f}, "
                     f"p50 {p[0]:.0f}, p90 {p[1]:.0f}, p99 {p[2]:.0f}, max {n.max()}")
    if args.sample and args.tokenizer:
        tok = FastTokenizer.from_file(args.tokenizer)
        first = m.shards_for("train")[0]
        tokens = read_tokens(f"{args.directory}/{first.name}")
        ends = np.flatnonzero(tokens == m.eos_id)
        start = 0
        for end in ends[: args.sample]:
            text = tok.decode(tokens[start:end].tolist())
            lines.append(f"--- doc ({end - start} tokens) ---\n{text[:300]}")
            start = end + 1
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
