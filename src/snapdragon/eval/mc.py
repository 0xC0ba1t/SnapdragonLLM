"""Multiple-choice evaluation by likelihood (the lm-evaluation-harness / HellaSwag method).

Each choice is appended to the context and scored by the model's log-likelihood of the *choice tokens
only*. ``acc`` picks the choice with the highest total log-likelihood; ``acc_norm`` divides by the
choice's length in tokens first (so longer answers are not penalised just for being longer).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from snapdragon.tokenizer.fast import FastTokenizer


@dataclass
class MCExample:
    context: str
    choices: list[str]
    label: int


def render_mc(tok: FastTokenizer, ex: MCExample, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """tokens (n_choices, L) and mask (n_choices, L); mask is 1 on choice tokens. Rows are right-padded
    with the pad id. If ``max_len`` is given, the *start* of the context is truncated to fit."""
    ctx = tok.encode(ex.context)
    rows, masks = [], []
    for choice in ex.choices:
        ans = tok.encode(choice)
        c = ctx
        if max_len is not None and len(c) + len(ans) > max_len:
            c = c[len(c) + len(ans) - max_len :]
        rows.append(c + ans)
        masks.append([0] * len(c) + [1] * len(ans))
    width = max(len(r) for r in rows)
    tokens = torch.full((len(rows), width), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        tokens[i, : len(r)] = torch.tensor(r)
        mask[i, : len(m)] = torch.tensor(m)
    return tokens, mask


@torch.no_grad()
def choice_losses(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-choice (sum of token losses, token count) over the masked (choice) region."""
    shift_logits = logits[:, :-1].float()
    targets = tokens[:, 1:]
    m = mask[:, 1:].float()  # mask shifted: position t predicts token t+1
    losses = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), targets.reshape(-1), reduction="none")
    losses = losses.view(targets.shape) * m
    return losses.sum(dim=1), m.sum(dim=1)


def predict(sum_loss: torch.Tensor, count: torch.Tensor) -> tuple[int, int]:
    """(argmin total loss, argmin length-normalised loss)."""
    return int(sum_loss.argmin()), int((sum_loss / count.clamp(min=1)).argmin())


@torch.no_grad()
def evaluate_mc(
    model,
    tok: FastTokenizer,
    examples: Iterable[MCExample],
    device: torch.device | str = "cpu",
    max_len: int | None = None,
    autocast=None,
) -> dict:
    """Accuracy and normalised accuracy of ``model`` (a CausalLM returning logits without targets)."""
    model.eval()
    n = correct = correct_norm = 0
    for ex in examples:
        tokens, mask = render_mc(tok, ex, max_len)
        tokens, mask = tokens.to(device), mask.to(device)
        ctx = autocast if autocast is not None else torch.autocast("cpu", enabled=False)
        with ctx:
            logits = model(tokens)
        sum_loss, count = choice_losses(logits, tokens, mask)
        p, p_norm = predict(sum_loss, count)
        n += 1
        correct += p == ex.label
        correct_norm += p_norm == ex.label
    return {"n": n, "acc": correct / max(n, 1), "acc_norm": correct_norm / max(n, 1)}
