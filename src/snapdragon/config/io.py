"""Reading and writing configs as YAML."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from snapdragon.config.loader import from_dict
from snapdragon.config.overrides import apply_overrides
from snapdragon.config.schema import Config


def _plain(obj: Any) -> Any:
    """Make asdict() output YAML-friendly (tuples become lists)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return _plain(asdict(cfg))


def dump_config(cfg: Config, path: str | Path | None = None) -> str:
    text = yaml.safe_dump(config_to_dict(cfg), sort_keys=False)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def deep_merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def read_yaml(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML mapping. An ``extends: other.yaml`` key (relative path) is merged underneath."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"circular extends: {' -> '.join(map(str, (*_seen, path)))}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a config file must be a mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return deep_merge(read_yaml(path.parent / parent, (*_seen, path)), data)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML config, apply ``section.key=value`` overrides, and validate it."""
    return from_dict(Config, apply_overrides(read_yaml(path), overrides))
