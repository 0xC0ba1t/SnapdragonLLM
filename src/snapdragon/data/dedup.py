"""Exact document de-duplication by normalized-content hash."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def doc_hash(text: str) -> int:
    digest = hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class ExactDedup:
    """Remembers 64-bit hashes of documents seen so far (collision odds are negligible at 1e9 docs
    for a quality filter; this is not a security boundary)."""

    def __init__(self) -> None:
        self._seen: set[int] = set()
        self.checked = 0
        self.dropped = 0

    def is_duplicate(self, text: str) -> bool:
        """True if an equivalent document was already seen; otherwise records this one."""
        self.checked += 1
        h = doc_hash(text)
        if h in self._seen:
            self.dropped += 1
            return True
        self._seen.add(h)
        return False

    def __len__(self) -> int:
        return len(self._seen)

    def save(self, path: str | Path) -> None:
        np.array(sorted(self._seen), dtype=np.uint64).tofile(path)

    @classmethod
    def load(cls, path: str | Path) -> ExactDedup:
        d = cls()
        d._seen = set(np.fromfile(path, dtype=np.uint64).tolist())
        return d
