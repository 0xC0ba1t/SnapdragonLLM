"""Convert common public instruction-tuning formats to ``{"messages": [...]}``."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

ROLE_ALIASES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
                "system": "system"}


def from_alpaca(row: dict) -> list[dict[str, str]]:
    """``{"instruction", "input", "output"}``: the input (if any) goes after the instruction."""
    prompt = row["instruction"].strip()
    if row.get("input", "").strip():
        prompt += "\n\n" + row["input"].strip()
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": row["output"].strip()}]


def from_sharegpt(row: dict) -> list[dict[str, str]]:
    """``{"conversations": [{"from": "human"|"gpt"|"system", "value": ...}]}``."""
    out = []
    for turn in row["conversations"]:
        role = ROLE_ALIASES.get(turn["from"].lower())
        if role is None:
            raise ValueError(f"unknown sharegpt speaker {turn['from']!r}")
        out.append({"role": role, "content": turn["value"].strip()})
    return out


def from_messages(row: dict) -> list[dict[str, str]]:
    """Already in chat form (OpenAI style), with role aliases normalised."""
    return [{"role": ROLE_ALIASES.get(m["role"].lower(), m["role"]), "content": m["content"].strip()}
            for m in row["messages"]]


CONVERTERS: dict[str, Callable[[dict], list[dict[str, str]]]] = {
    "alpaca": from_alpaca, "sharegpt": from_sharegpt, "messages": from_messages,
}


def normalise(rows: Iterable[dict], fmt: str) -> Iterator[dict]:
    """Yield ``{"messages": ...}`` rows, dropping conversations that are unusable for SFT (nothing the
    assistant says, empty text, or that do not alternate sensibly after the optional system turn)."""
    if fmt not in CONVERTERS:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(CONVERTERS)}")
    convert = CONVERTERS[fmt]
    for row in rows:
        messages = convert(row)
        if not messages or any(not m["content"] for m in messages):
            continue
        body = [m for m in messages if m["role"] != "system"]
        if not body or body[0]["role"] != "user" or body[-1]["role"] != "assistant":
            continue
        if any(a["role"] == b["role"] for a, b in zip(body, body[1:])):
            continue
        yield {"messages": messages}
