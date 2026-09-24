"""Rebuild a trained model from a checkpoint directory (for evaluation, generation, export)."""
from __future__ import annotations

from pathlib import Path

import torch

from snapdragon.checkpoint.io import latest_checkpoint, load_checkpoint
from snapdragon.config.loader import from_dict
from snapdragon.config.schema import Config
from snapdragon.model.factory import build_model
from snapdragon.model.lm import CausalLM


def resolve_checkpoint(path: str | Path) -> Path:
    """A checkpoint directory, or a run's ``checkpoints`` directory (newest checkpoint is used)."""
    path = Path(path)
    if (path / "model.pt").exists():
        return path
    latest = latest_checkpoint(path)
    if latest is None:
        raise FileNotFoundError(f"no complete checkpoint found in {path}")
    return latest


def load_model_from_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[CausalLM, Config]:
    """Build the architecture recorded in the checkpoint and load its weights. Returns (model, config)."""
    ckpt = load_checkpoint(resolve_checkpoint(path), map_location="cpu", only=["model", "meta"])
    cfg = from_dict(Config, ckpt["meta"]["config"])
    model = build_model(cfg, device="meta")
    model.to_empty(device=device)
    model.load_state_dict(ckpt["model"])
    model.transformer.rope.rebuild(device)  # rotary tables are buffers, not part of the state dict
    return model.eval(), cfg
