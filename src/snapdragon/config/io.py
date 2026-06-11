"""Reading and writing configs as YAML."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from snapdragon.config.loader import from_dict
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


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level of a config file must be a mapping")
    return data


def load_config(path: str | Path) -> Config:
    return from_dict(Config, read_yaml(path))
