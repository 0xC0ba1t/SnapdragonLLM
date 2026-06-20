"""On-disk shard format.

A shard is two files sharing a stem:

* ``<stem>.bin``  raw little-endian uint16 token ids (every document is followed by <|eos|>)
* ``<stem>.idx``  raw little-endian int64 document start offsets, plus the total length at the end

uint16 is enough because the vocabulary has 32,768 ids.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

TOKEN_DTYPE = np.dtype("<u2")
INDEX_DTYPE = np.dtype("<i8")
MAX_VOCAB = 1 << 16


class ShardWriter:
    """Accumulates documents and writes fixed-size shards (the last one may be smaller)."""

    def __init__(self, out_dir: str | Path, prefix: str, shard_tokens: int, eos_id: int) -> None:
        if shard_tokens < 2:
            raise ValueError("shard_tokens must be at least 2")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.eos_id = eos_id
        self.files: list[tuple[str, int, int]] = []  # (stem, tokens, docs)
        self._buf: list[np.ndarray] = []
        self._buf_tokens = 0
        self._docs: list[int] = []  # start offsets within the current shard

    def add_document(self, token_ids: Iterable[int]) -> None:
        arr = np.fromiter(token_ids, dtype=np.int64)
        if arr.size and (arr.min() < 0 or arr.max() >= MAX_VOCAB):
            raise ValueError("token ids must fit in uint16")
        doc = np.concatenate([arr, [self.eos_id]]).astype(TOKEN_DTYPE)
        self._docs.append(self._buf_tokens)
        self._buf.append(doc)
        self._buf_tokens += doc.size
        if self._buf_tokens >= self.shard_tokens:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        stem = f"{self.prefix}_{len(self.files):05d}"
        tokens = np.concatenate(self._buf)
        tokens.tofile(self.out_dir / f"{stem}.bin")
        np.array([*self._docs, tokens.size], dtype=INDEX_DTYPE).tofile(self.out_dir / f"{stem}.idx")
        self.files.append((stem, int(tokens.size), len(self._docs)))
        self._buf, self._buf_tokens, self._docs = [], 0, []

    def close(self) -> list[tuple[str, int, int]]:
        self._flush()
        return self.files

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_tokens(stem_path: str | Path) -> np.ndarray:
    return np.fromfile(Path(stem_path).with_suffix(".bin"), dtype=TOKEN_DTYPE)


def read_index(stem_path: str | Path) -> np.ndarray:
    """Document start offsets followed by the total token count."""
    return np.fromfile(Path(stem_path).with_suffix(".idx"), dtype=INDEX_DTYPE)
