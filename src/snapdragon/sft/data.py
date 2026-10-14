"""Chat datasets for supervised fine-tuning: jsonl -> (token ids, assistant-only loss mask)."""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from snapdragon.tokenizer.chat import ROLE_TOKENS, encode_chat
from snapdragon.tokenizer.fast import FastTokenizer


@dataclass
class SFTExample:
    ids: list[int]
    mask: list[int]  # 1 where the loss applies (assistant content + its end-of-turn token)

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.mask):
            raise ValueError("ids and mask must have the same length")


def read_messages(path: str | Path) -> Iterator[list[dict[str, str]]]:
    """Each jsonl line is ``{"messages": [{"role": ..., "content": ...}, ...]}``."""
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{n}: expected a non-empty 'messages' list")
            for m in messages:
                if m.get("role") not in ROLE_TOKENS or not isinstance(m.get("content"), str):
                    raise ValueError(f"{path}:{n}: bad message {m!r}")
            yield messages


def tokenize_chats(
    tok: FastTokenizer, chats: Iterable[list[dict[str, str]]], max_len: int | None = None
) -> Iterator[SFTExample]:
    """Tokenize conversations; those with no assistant tokens, or longer than ``max_len``, are skipped
    (truncating a conversation could cut an answer in half and teach the model to stop mid-sentence)."""
    for messages in chats:
        ids, mask = encode_chat(tok, messages)
        if not any(mask):
            continue
        if max_len is not None and len(ids) > max_len:
            continue
        yield SFTExample(ids, mask)
