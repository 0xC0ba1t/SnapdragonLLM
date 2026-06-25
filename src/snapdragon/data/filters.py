"""Cheap heuristic document-quality filters applied before tokenization."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualityFilter:
    min_chars: int = 200
    max_chars: int = 200_000
    min_alpha_ratio: float = 0.6  # letters / non-space characters
    max_dup_line_ratio: float = 0.3  # share of lines that repeat an earlier line
    min_mean_word_len: float = 3.0
    max_mean_word_len: float = 10.0

    def reason_to_drop(self, text: str) -> str | None:
        """Name of the first failed check, or None if the document is kept."""
        n = len(text)
        if n < self.min_chars:
            return "too_short"
        if n > self.max_chars:
            return "too_long"
        non_space = [c for c in text if not c.isspace()]
        if not non_space:
            return "empty"
        if sum(c.isalpha() for c in non_space) / len(non_space) < self.min_alpha_ratio:
            return "low_alpha"
        words = text.split()
        mean_len = sum(map(len, words)) / len(words)
        if not self.min_mean_word_len <= mean_len <= self.max_mean_word_len:
            return "word_length"
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 3 and 1 - len(set(lines)) / len(lines) > self.max_dup_line_ratio:
            return "repeated_lines"
        return None

    def keep(self, text: str) -> bool:
        return self.reason_to_drop(text) is None
