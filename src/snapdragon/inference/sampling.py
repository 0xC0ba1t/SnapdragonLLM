"""Next-token sampling: temperature, top-k, top-p (nucleus), min-p and repetition penalty."""
from __future__ import annotations

from dataclasses import dataclass

import torch

NEG_INF = float("-inf")


@dataclass
class SamplingConfig:
    temperature: float = 0.8  # 0 means greedy (argmax)
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    min_p: float = 0.0  # 0 disables
    repetition_penalty: float = 1.0  # 1.0 disables

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError("min_p must be in [0, 1)")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k highest logits per row (ties at the threshold are kept)."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, NEG_INF)


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probability mass reaches p (always at least one)."""
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    drop = (cumulative - probs) >= p  # mass *before* this token already reached p
    sorted_logits = sorted_logits.masked_fill(drop, NEG_INF)
    return torch.full_like(logits, NEG_INF).scatter(-1, order, sorted_logits)


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Drop tokens whose probability is below ``min_p`` times the most likely token's probability.
    Unlike top-p the cutoff adapts to confidence: sharp distributions keep few tokens, flat ones many."""
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    floor = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < floor, NEG_INF)


def apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL-style: tokens already in ``history`` (B, S) get their logit divided (if positive) or
    multiplied (if negative) by ``penalty``, so they become less likely either way."""
    if penalty == 1.0 or history.numel() == 0:
        return logits
    seen = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, history, True)
    penalised = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalised, logits)


def sample_next(
    logits: torch.Tensor,
    cfg: SamplingConfig,
    generator: torch.Generator | None = None,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    """logits (B, V) -> next token ids (B,). ``history`` (B, S) feeds the repetition penalty."""
    logits = logits.float()
    if history is not None:
        logits = apply_repetition_penalty(logits, history, cfg.repetition_penalty)
    if cfg.greedy:
        return logits.argmax(dim=-1)
    logits = logits / cfg.temperature
    logits = apply_top_k(logits, cfg.top_k)
    logits = apply_top_p(logits, cfg.top_p)
    logits = apply_min_p(logits, cfg.min_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
