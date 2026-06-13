"""Pre-tokenization: split text into 'words' that BPE merges never cross."""
from __future__ import annotations

import regex

# Contractions, letters, numbers (grouped by 1-3 digits), punctuation runs, and whitespace.
# Splitting digits into short groups keeps numbers from becoming rare monolithic tokens.
PATTERN = regex.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
    r"""|[^\r\n\p{L}\p{N}]?\p{L}+"""
    r"""|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*"""
    r"""|\s*[\r\n]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


def pretokenize(text: str) -> list[str]:
    return PATTERN.findall(text)
