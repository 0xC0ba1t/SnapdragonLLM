"""Chat formatting: <|bos|> then, per turn, <|role|> content <|end_turn|>."""
from __future__ import annotations

from collections.abc import Sequence

from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.tokenizer.special_tokens import ASSISTANT, BOS, END_TURN, SYSTEM, USER

ROLE_TOKENS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}

Message = dict[str, str]


def _check(messages: Sequence[Message]) -> None:
    for m in messages:
        if m.get("role") not in ROLE_TOKENS:
            raise ValueError(f"unknown role {m.get('role')!r}; expected one of {list(ROLE_TOKENS)}")
        if not isinstance(m.get("content"), str):
            raise ValueError("every message needs a string 'content'")


def render_chat(messages: Sequence[Message], add_generation_prompt: bool = False) -> str:
    """Render to text; special tokens are written literally and parsed back by the tokenizer."""
    _check(messages)
    out = [BOS]
    for m in messages:
        out.append(f"{ROLE_TOKENS[m['role']]}{m['content']}{END_TURN}")
    if add_generation_prompt:
        out.append(ASSISTANT)
    return "".join(out)


def encode_chat(
    tok: FastTokenizer, messages: Sequence[Message], add_generation_prompt: bool = False
) -> tuple[list[int], list[int]]:
    """Return (token ids, loss mask). The mask is 1 on assistant content and its end-of-turn."""
    _check(messages)
    ids: list[int] = [tok.bos_id]
    mask: list[int] = [0]
    for m in messages:
        role_id = tok.encode(ROLE_TOKENS[m["role"]])
        body = tok.encode(m["content"])
        end = tok.encode(END_TURN)
        train = 1 if m["role"] == "assistant" else 0
        ids += role_id + body + end
        mask += [0] * len(role_id) + [train] * (len(body) + len(end))
    if add_generation_prompt:
        gen = tok.encode(ASSISTANT)
        ids += gen
        mask += [0] * len(gen)
    return ids, mask
