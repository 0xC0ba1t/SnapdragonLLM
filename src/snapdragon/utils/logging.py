"""Rank-aware logging helpers."""
from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def get_rank() -> int:
    """Global rank from torchrun-style env vars (0 when not distributed)."""
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_logging(level: int | str = logging.INFO, force: bool = False) -> None:
    """Configure the root logger once. Non-main ranks only log warnings and above."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    effective = level if is_main_process() else logging.WARNING
    logging.basicConfig(level=effective, format=_FORMAT, stream=sys.stderr, force=True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
