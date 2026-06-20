"""Helpers for packed sequences: documents are concatenated, separated by <|eos|>.

All functions take integer tensors of shape (B, T) holding the *input* tokens.
"""
from __future__ import annotations

import torch

IGNORE_INDEX = -100


def doc_ids_from_tokens(x: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Document index of every position; an <|eos|> belongs to the document it terminates.

    ``[a b eos c d eos e]`` -> ``[0 0 0 1 1 1 2]``.
    """
    is_eos = (x == eos_id).to(torch.int32)
    return torch.cumsum(is_eos, dim=1) - is_eos


def position_ids_from_tokens(x: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Position inside the current document, restarting at 0 after each <|eos|>.

    ``[a b eos c d eos e]`` -> ``[0 1 2 0 1 2 0]``.
    """
    _, T = x.shape
    idx = torch.arange(T, device=x.device).expand_as(x)
    is_start = torch.zeros_like(x, dtype=torch.bool)
    is_start[:, 0] = True
    is_start[:, 1:] = x[:, :-1] == eos_id
    start = torch.where(is_start, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return idx - start


def split_window(window: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Cut a window of ``max_len + 1`` tokens into ``max_len // seq_len`` windows of ``seq_len + 1``.

    Neighbouring pieces share one boundary token (the last target of one is the first input of the
    next), so no token is wasted when training with a shorter sequence length.
    """
    max_len = window.shape[-1] - 1
    if max_len % seq_len:
        raise ValueError(f"seq_len {seq_len} must divide the window length {max_len}")
    pieces = [window[..., k * seq_len : k * seq_len + seq_len + 1] for k in range(max_len // seq_len)]
    return torch.stack(pieces, dim=-2).reshape(-1, seq_len + 1) if window.dim() > 1 else torch.stack(pieces)
