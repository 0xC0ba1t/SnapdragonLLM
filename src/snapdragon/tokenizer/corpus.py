"""Text sources for tokenizer training: local .txt/.jsonl files or a streamed HF dataset."""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def iter_local_texts(paths: Iterable[str | Path], field: str = "text") -> Iterator[str]:
    """Yield documents: one per non-empty line of .jsonl (field ``field``), or one per .txt file."""
    for p in map(Path, paths):
        if p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)[field]
        else:
            yield p.read_text(encoding="utf-8")


def iter_hf_texts(name: str, config: str | None = None, split: str = "train", field: str = "text") -> Iterator[str]:
    """Stream a HuggingFace dataset (needs the optional ``datasets`` dependency)."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - exercised only without the extra installed
        raise ImportError("install the 'data' extra: pip install -e .[data]") from e
    ds = load_dataset(name, config, split=split, streaming=True)
    for row in ds:
        yield row[field]


def take_bytes(texts: Iterable[str], max_bytes: int) -> Iterator[str]:
    """Stop after roughly ``max_bytes`` of UTF-8 text (the document that crosses it is kept)."""
    seen = 0
    for t in texts:
        yield t
        seen += len(t.encode("utf-8"))
        if seen >= max_bytes:
            return
