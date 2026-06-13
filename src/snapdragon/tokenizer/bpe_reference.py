"""A small, readable byte-level BPE (train + encode + decode).

This is the educational/reference implementation that the tests use as an oracle. Real
training runs use the fast HuggingFace-backed tokenizer in ``fast.py`` with the same design.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

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

