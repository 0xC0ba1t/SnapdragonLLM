"""Special tokens. They occupy the last ids of the vocabulary so BPE merges keep stable ids."""
from __future__ import annotations

BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END_TURN = "<|end_turn|>"

SPECIAL_TOKENS: tuple[str, ...] = (BOS, EOS, PAD, SYSTEM, USER, ASSISTANT, END_TURN)


def special_token_ids(vocab_size: int) -> dict[str, int]:
    """Specials are appended at the end: ids vocab_size-len(SPECIAL) .. vocab_size-1."""
    first = vocab_size - len(SPECIAL_TOKENS)
    if first < 256:
        raise ValueError(f"vocab_size {vocab_size} is too small for 256 bytes + specials")
    return {tok: first + i for i, tok in enumerate(SPECIAL_TOKENS)}
