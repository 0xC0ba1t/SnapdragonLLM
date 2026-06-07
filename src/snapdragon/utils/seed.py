"""Reproducibility helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, rank: int = 0) -> int:
    """Seed python, numpy and torch.

    The rank is mixed in so ranks get distinct streams for dropout and sampling;
    model init should use the unmixed seed so every rank builds identical weights.
    """
    s = seed + rank
    random.seed(s)
    np.random.seed(s)  # noqa: NPY002 - legacy global state is intentionally seeded too
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


def get_rng_state() -> dict:
    """Capture all RNG states (for checkpointing)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
