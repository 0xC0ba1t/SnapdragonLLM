"""Tokenizer quality metrics."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from snapdragon.tokenizer.fast import FastTokenizer


@dataclass
class Fertility:
    tokens: int
    words: int
    bytes: int

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / max(self.words, 1)

    @property
    def bytes_per_token(self) -> float:
        """Compression: higher is better (more text per token)."""
        return self.bytes / max(self.tokens, 1)


def measure_fertility(tok: FastTokenizer, texts: Iterable[str]) -> Fertility:
    tokens = words = nbytes = 0
    for text in texts:
        tokens += len(tok.encode(text))
        words += len(text.split())
        nbytes += len(text.encode("utf-8"))
    return Fertility(tokens=tokens, words=words, bytes=nbytes)
