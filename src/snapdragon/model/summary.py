"""Human-readable parameter breakdown."""
from __future__ import annotations

from collections import OrderedDict

from torch import nn


def parameter_breakdown(model: nn.Module) -> OrderedDict[str, int]:
    """Parameters grouped by role: embeddings, attention, mlp, norms, other (tied weights once)."""
    groups: OrderedDict[str, int] = OrderedDict((k, 0) for k in ("embeddings", "attention", "mlp", "norms", "other"))
    for name, p in model.named_parameters():
        if "embed" in name or "lm_head" in name:
            key = "embeddings"
        elif ".attn." in name and "norm" not in name.split(".attn.")[1].split(".")[0]:
            key = "attention"
        elif ".attn.q_norm" in name or ".attn.k_norm" in name or "norm" in name.split(".")[-2]:
            key = "norms"
        elif ".mlp." in name:
            key = "mlp"
        else:
            key = "other"
        groups[key] += p.numel()
    return groups


def format_summary(model: nn.Module) -> str:
    groups = parameter_breakdown(model)
    total = sum(groups.values())
    lines = [f"{'group':<12}{'params':>14}{'share':>8}"]
    for k, v in groups.items():
        if v:
            lines.append(f"{k:<12}{v:>14,}{v / total:>8.1%}")
    lines.append(f"{'total':<12}{total:>14,}")
    return "\n".join(lines)
