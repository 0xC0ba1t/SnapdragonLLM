"""One optimizer facade over several (Muon for matrices, AdamW for the rest)."""
from __future__ import annotations

import torch


class HybridOptimizer:
    """Presents the torch.optim.Optimizer surface the trainer and LR schedule rely on:
    ``param_groups`` (the very same dict objects, so schedulers mutate the real groups), ``step``,
    ``zero_grad`` and a combined ``state_dict`` / ``load_state_dict``."""

    def __init__(self, **optimizers: torch.optim.Optimizer) -> None:
        if not optimizers:
            raise ValueError("HybridOptimizer needs at least one optimizer")
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict]:
        return [g for opt in self.optimizers.values() for g in opt.param_groups]

    def step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state: dict) -> None:
        missing = set(self.optimizers) - set(state)
        extra = set(state) - set(self.optimizers)
        if missing or extra:
            raise ValueError(f"optimizer state mismatch: missing {sorted(missing)}, unexpected {sorted(extra)}")
        for name, opt in self.optimizers.items():
            opt.load_state_dict(state[name])
