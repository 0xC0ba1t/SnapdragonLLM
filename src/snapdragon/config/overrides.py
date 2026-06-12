"""Dotted command line overrides, e.g. ``train.micro_batch_size=2 optim.adam_lr=1e-3``."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import yaml


def parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"override {item!r} must look like section.key=value")
    key, raw = item.split("=", 1)
    parts = key.strip().split(".")
    if not all(parts):
        raise ValueError(f"override {item!r} has an empty path segment")
    return parts, yaml.safe_load(raw)  # YAML scalars: 3e-4 is a float only as 3.0e-4, handled below


def _coerce_number(value: Any, raw: str) -> Any:
    # PyYAML reads "3e-4" as a string (it wants a dot); accept it as a float like users expect
    if isinstance(value, str):
        try:
            return float(raw)
        except ValueError:
            return value
    return value


def apply_overrides(data: dict[str, Any], items: Iterable[str]) -> dict[str, Any]:
    """Return a copy of ``data`` with each ``a.b.c=value`` applied (intermediate dicts created)."""
    out = _deepcopy(data)
    for item in items:
        parts, value = parse_override(item)
        value = _coerce_number(value, item.split("=", 1)[1])
        node = out
        for p in parts[:-1]:
            nxt = node.setdefault(p, {})
            if not isinstance(nxt, dict):
                raise ValueError(f"override {item!r}: {p!r} is not a section")
            node = nxt
        node[parts[-1]] = value
    return out


def _deepcopy(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _deepcopy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deepcopy(v) for v in d]
    return d
