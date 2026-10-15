"""Pack variable-length chat examples into fixed-length rows so no compute is spent on padding."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from snapdragon.data.packing import IGNORE_INDEX
from snapdragon.sft.data import SFTExample


@dataclass
class PackedRows:
    x: torch.Tensor  # (R, T) input tokens
    y: torch.Tensor  # (R, T) targets, IGNORE_INDEX wherever the loss must not apply
    position_ids: torch.Tensor  # (R, T) positions restarting at every example


def pack_examples(examples: Iterable[SFTExample], seq_len: int, pad_id: int) -> PackedRows:
    """Greedy sequential packing into rows of ``seq_len + 1`` tokens (inputs plus the shifted target).

    Examples are never split across rows; an example longer than a row is an error (filter with
    ``tokenize_chats(max_len=seq_len + 1)``). Targets are -100 on padding and on every non-assistant
    token, including the first token of each example, so one example never trains on predicting the next.
    """
    width = seq_len + 1
    rows: list[tuple[list[int], list[int], list[int]]] = []
    ids: list[int] = []
    mask: list[int] = []
    pos: list[int] = []

    def flush() -> None:
        nonlocal ids, mask, pos
        if ids:
            pad = width - len(ids)
            rows.append((ids + [pad_id] * pad, mask + [0] * pad, pos + [0] * pad))
        ids, mask, pos = [], [], []

    for ex in examples:
        n = len(ex.ids)
        if n > width:
            raise ValueError(f"example of {n} tokens does not fit a row of {width}")
        if len(ids) + n > width:
            flush()
        ids += ex.ids
        mask += ex.mask
        pos += list(range(n))
    flush()
    if not rows:
        return PackedRows(*(torch.empty(0, seq_len, dtype=torch.long) for _ in range(3)))
    t_ids = torch.tensor([r[0] for r in rows])
    t_mask = torch.tensor([r[1] for r in rows])
    t_pos = torch.tensor([r[2] for r in rows])
    y = torch.where(t_mask[:, 1:] == 1, t_ids[:, 1:], torch.full_like(t_ids[:, 1:], IGNORE_INDEX))
    return PackedRows(t_ids[:, :-1].contiguous(), y.contiguous(), t_pos[:, :-1].contiguous())
