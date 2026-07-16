"""Decide which optimizer handles which parameter."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ParamSplit:
    muon: list[nn.Parameter]  # hidden 2-D weight matrices inside the transformer blocks
    adam_decay: list[nn.Parameter]  # embeddings / output matrix: AdamW with weight decay
    adam_no_decay: list[nn.Parameter]  # gains, biases and other vectors: AdamW, no decay
    names: dict[str, str]  # parameter name -> group, for logging and tests

    def counts(self) -> dict[str, int]:
        return {
            "muon": sum(p.numel() for p in self.muon),
            "adam_decay": sum(p.numel() for p in self.adam_decay),
            "adam_no_decay": sum(p.numel() for p in self.adam_no_decay),
        }


def split_params(model: nn.Module) -> ParamSplit:
    """Muon orthogonalises 2-D updates, which suits hidden matrices but not embedding tables
    (rows are independent token vectors) or 1-D gains, which go to AdamW."""
    muon, decay, no_decay, names = [], [], [], {}
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2 and ".blocks." in f".{name}":
            muon.append(p)
            names[name] = "muon"
        elif p.ndim >= 2:
            decay.append(p)
            names[name] = "adam_decay"
        else:
            no_decay.append(p)
            names[name] = "adam_no_decay"
    return ParamSplit(muon, decay, no_decay, names)


def assert_partition(model: nn.Module, split: ParamSplit) -> None:
    """Every trainable parameter must be owned by exactly one group."""
    owned = [id(p) for g in (split.muon, split.adam_decay, split.adam_no_decay) for p in g]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(owned) != len(set(owned)) or set(owned) != expected:
        raise AssertionError("parameter groups do not partition the trainable parameters")


def numel_by_group(split: ParamSplit) -> dict[str, int]:
    return split.counts()


__all__ = ["ParamSplit", "assert_partition", "numel_by_group", "split_params", "torch"]
