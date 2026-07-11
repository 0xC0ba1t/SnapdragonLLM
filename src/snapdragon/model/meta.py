"""Meta-device construction: build the structure for free, allocate and initialise later."""
from __future__ import annotations

import torch

from snapdragon.model.lm import CausalLM


def materialize(model: CausalLM, device: torch.device | str = "cpu") -> CausalLM:
    """Turn a meta-device model into a real, freshly initialised one on ``device``.

    ``to_empty`` allocates uninitialised storage, so parameters are re-initialised and the rotary
    tables rebuilt. Seeding before this call gives the same weights as a direct construction.
    """
    model.to_empty(device=device)
    model.transformer.reset_parameters()
    model.transformer.rope.rebuild(device)
    return model
