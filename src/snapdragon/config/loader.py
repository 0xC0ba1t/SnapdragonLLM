"""Strict conversion of plain dicts (e.g. parsed YAML) into the config dataclasses."""
from __future__ import annotations

import difflib
import types
import typing
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar, get_type_hints

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised for unknown keys or values of the wrong type, with the offending key path."""


def from_dict(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or '<root>'}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = [f.name for f in fields(cls)]
    for key in data:
        if key not in known:
            close = difflib.get_close_matches(key, known, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise ConfigError(f"{_join(path, key)}: unknown key{hint}")
    kwargs = {k: _convert(hints[k], v, _join(path, k)) for k, v in data.items()}
    return cls(**kwargs)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _convert(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _convert(args[0], value, path)
    if is_dataclass(tp):
        return from_dict(tp, value, path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (inner,) = typing.get_args(tp)
        return [_convert(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a sequence, got {type(value).__name__}")
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(_convert(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        _, vt = typing.get_args(tp)
        return {k: _convert(vt, v, f"{path}.{k}") for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value
