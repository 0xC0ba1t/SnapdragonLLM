"""A small, readable byte-level BPE (train + encode + decode).

This is the educational/reference implementation that the tests use as an oracle. Real
training runs use the fast HuggingFace-backed tokenizer in ``fast.py`` with the same design.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from snapdragon.tokenizer.pretokenize import pretokenize
from snapdragon.tokenizer.special_tokens import SPECIAL_TOKENS

Pair = tuple[int, int]


def count_words(texts: Iterable[str]) -> Counter[tuple[int, ...]]:
    """Frequency table of pre-tokenized words, each as a tuple of byte values."""
    words: Counter[tuple[int, ...]] = Counter()
    for text in texts:
        for w in pretokenize(text):
            words[tuple(w.encode("utf-8"))] += 1
    return words


def _pair_counts(words: dict[tuple[int, ...], int]) -> Counter[Pair]:
    pairs: Counter[Pair] = Counter()
    for word, freq in words.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs


def _merge_word(word: tuple[int, ...], pair: Pair, new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_merges(words: Counter[tuple[int, ...]], num_merges: int) -> list[Pair]:
    """Greedy BPE: repeatedly merge the most frequent adjacent pair.

    Ties break toward the numerically smallest pair so training is deterministic.
    """
    words = dict(words)
    merges: list[Pair] = []
    for step in range(num_merges):
        pairs = _pair_counts(words)
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        new_id = 256 + step
        merges.append(best)
        words = _apply_merge(words, best, new_id)
    return merges


def _apply_merge(words: dict[tuple[int, ...], int], pair: Pair, new_id: int):
    merged: dict[tuple[int, ...], int] = {}
    for word, freq in words.items():
        key = _merge_word(word, pair, new_id) if len(word) > 1 else word
        merged[key] = merged.get(key, 0) + freq
    return merged


def num_merges_for(vocab_size: int) -> int:
    return vocab_size - 256 - len(SPECIAL_TOKENS)


class BPETokenizer:
    """Encode/decode given an ordered merge list (rank = position in the list)."""

    def __init__(self, merges: list[Pair]) -> None:
        self.merges = [tuple(m) for m in merges]
        self.ranks: dict[Pair, int] = {m: i for i, m in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> BPETokenizer:
        return cls(train_merges(count_words(texts), num_merges_for(vocab_size)))

    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            ids = list(_merge_word(tuple(ids), best_pair, 256 + best_rank))
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for word in pretokenize(text):
            out.extend(self._encode_word(word))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def save(self, path: str | Path) -> None:
        payload = {"type": "bpe-reference", "version": 1, "merges": [list(m) for m in self.merges]}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "bpe-reference":
            raise ValueError(f"{path}: not a reference BPE file")
        return cls([tuple(m) for m in payload["merges"]])
