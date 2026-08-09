"""Checkpoints are directories ``step_00000500/`` holding one ``<name>.pt`` file per component.

A checkpoint is written into a temporary sibling directory and renamed into place only after every
file is on disk and a ``COMPLETE`` marker exists, so a crash mid-save can never leave something that
looks valid but is truncated.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch

COMPLETE = "COMPLETE"
_STEP_RE = re.compile(r"^step_(\d{8})$")


def step_dir_name(step: int) -> str:
    return f"step_{step:08d}"


def save_checkpoint(root: str | Path, step: int, components: dict[str, Any]) -> Path:
    """Write each component with torch.save and publish the directory atomically."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / step_dir_name(step)
    tmp = root / f".{final.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        for name, obj in components.items():
            torch.save(obj, tmp / f"{name}.pt")
        (tmp / COMPLETE).write_text(str(step))
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def list_checkpoints(root: str | Path) -> list[tuple[int, Path]]:
    """Complete checkpoints as (step, path), oldest first. Partial or foreign dirs are ignored."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for p in root.iterdir():
        m = _STEP_RE.match(p.name)
        if m and p.is_dir() and (p / COMPLETE).exists():
            found.append((int(m.group(1)), p))
    return sorted(found)


def latest_checkpoint(root: str | Path) -> Path | None:
    found = list_checkpoints(root)
    return found[-1][1] if found else None


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load every ``<name>.pt`` component of a checkpoint directory."""
    path = Path(path)
    if not (path / COMPLETE).exists():
        raise FileNotFoundError(f"{path} is not a complete checkpoint (missing {COMPLETE} marker)")
    return {
        f.stem: torch.load(f, map_location=map_location, weights_only=False)
        for f in sorted(path.glob("*.pt"))
    }


def prune_checkpoints(root: str | Path, keep_last_n: int) -> list[Path]:
    """Delete all but the newest ``keep_last_n`` complete checkpoints; returns what was removed."""
    if keep_last_n < 1:
        raise ValueError("keep_last_n must be at least 1")
    doomed = [p for _, p in list_checkpoints(root)[:-keep_last_n]]
    for p in doomed:
        shutil.rmtree(p)
    return doomed
