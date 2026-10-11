"""Save/load a model as ``model.safetensors`` + ``config.json`` (+ tokenizer): the release format.

safetensors is a plain tensor container: loading it cannot execute code (unlike pickled ``.pt``), it is
memory-mappable, and every serious inference stack reads it.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from snapdragon.config.loader import from_dict
from snapdragon.config.model import ModelConfig
from snapdragon.model.factory import build_model
from snapdragon.model.lm import CausalLM

WEIGHTS = "model.safetensors"
CONFIG = "config.json"
FORMAT_VERSION = 1


def save_pretrained(
    model: CausalLM, out_dir: str | Path, tokenizer_path: str | Path | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
) -> Path:
    """Write weights (cast to ``dtype``, ``None`` keeps the training dtype) and the architecture config."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: (v.to(dtype) if dtype else v).detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, out / WEIGHTS, metadata={"format": "pt", "snapdragon_format": str(FORMAT_VERSION)})
    config = {"architecture": "snapdragon", "format_version": FORMAT_VERSION, **asdict(model.cfg)}
    (out / CONFIG).write_text(json.dumps(config, indent=2), encoding="utf-8")
    if tokenizer_path is not None:
        shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    return out


def load_pretrained(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> CausalLM:
    """Rebuild the model recorded in ``config.json`` and load ``model.safetensors`` into it."""
    path = Path(path)
    raw = json.loads((path / CONFIG).read_text(encoding="utf-8"))
    if raw.pop("architecture", None) != "snapdragon":
        raise ValueError(f"{path / CONFIG} is not a snapdragon config")
    if raw.pop("format_version", None) != FORMAT_VERSION:
        raise ValueError("unsupported snapdragon export format version")
    cfg = from_dict(ModelConfig, raw)
    model = build_model(cfg, device="meta")
    model.to_empty(device=device)
    state = {k: v.to(dtype) for k, v in load_file(str(path / WEIGHTS)).items()}
    model.load_state_dict(state)
    model.to(dtype)
    model.transformer.rope.rebuild(device)
    return model.eval()
