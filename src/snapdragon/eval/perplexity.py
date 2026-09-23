"""Perplexity and bits-per-byte over a tokenized dataset split."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from snapdragon.data.memmap_dataset import TokenDataset
from snapdragon.data.val import ValLoader


@dataclass
class PerplexityResult:
    loss: float  # mean cross-entropy in nats per token
    tokens: int

    @property
    def ppl(self) -> float:
        return math.exp(min(self.loss, 50.0))

    @property
    def bits_per_token(self) -> float:
        return self.loss / math.log(2)

    def bits_per_byte(self, bytes_per_token: float) -> float:
        """Tokenizer-independent quality: comparable across different vocabularies."""
        return self.bits_per_token / bytes_per_token


@torch.no_grad()
def evaluate_perplexity(
    model, dataset: TokenDataset, seq_len: int, batch_size: int = 4, max_batches: int | None = None,
    device: torch.device | str = "cpu", autocast=None,
) -> PerplexityResult:
    """Mean CE over contiguous non-overlapping windows. The model is any CausalLM; the fused loss
    means logits are never materialised, so long evaluations use the same low memory as training."""
    model.eval()
    loader = ValLoader(dataset, seq_len, batch_size, max_batches)
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=False)
        ctx = autocast if autocast is not None else torch.autocast("cpu", enabled=False)
        with ctx:
            _, ce, _ = model(batch.x, batch.y)
        n = batch.y.numel()
        total += float(ce) * n
        count += n
    return PerplexityResult(loss=total / max(count, 1), tokens=count)


def bytes_per_token(tokenizer, texts) -> float:
    """Average UTF-8 bytes per token on a text sample (needed for bits-per-byte)."""
    nbytes = ntok = 0
    for t in texts:
        nbytes += len(t.encode("utf-8"))
        ntok += len(tokenizer.encode(t))
    return nbytes / max(ntok, 1)
