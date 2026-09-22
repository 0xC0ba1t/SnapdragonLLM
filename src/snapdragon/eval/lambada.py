"""LAMBADA: predict the final word of a passage. Accuracy needs *every* token of the word to be the
model's greedy choice; perplexity is over the final word's tokens."""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F

from snapdragon.eval.data import fetch
from snapdragon.tokenizer.fast import FastTokenizer

LAMBADA_URL = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"


def split_last_word(text: str) -> tuple[str, str]:
    """('the cat sat on the', ' mat') for 'the cat sat on the mat'."""
    context, _, word = text.strip().rpartition(" ")
    return context, " " + word


def load_lambada(path: str | Path | None = None, limit: int = 0) -> list[str]:
    texts = []
    with open(path or fetch(LAMBADA_URL), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
            if limit and len(texts) >= limit:
                break
    return texts


@torch.no_grad()
def evaluate_lambada(model, tok: FastTokenizer, texts: Iterable[str], device="cpu", max_len: int | None = None) -> dict:
    model.eval()
    n = correct = 0
    nll = ntok = 0.0
    for text in texts:
        ctx, word = split_last_word(text)
        ctx_ids, word_ids = tok.encode(ctx), tok.encode(word)
        ids = ctx_ids + word_ids
        if max_len is not None and len(ids) > max_len:
            ids = ids[-max_len:]
        x = torch.tensor([ids], device=device)
        logits = model(x)[0].float()
        k = len(word_ids)
        pred_logits = logits[-k - 1 : -1]  # position t predicts token t+1
        target = x[0, -k:]
        correct += bool((pred_logits.argmax(-1) == target).all())
        nll += F.cross_entropy(pred_logits, target, reduction="sum").item()
        ntok += k
        n += 1
    return {"n": n, "acc": correct / max(n, 1), "ppl": math.exp(nll / max(ntok, 1))}
