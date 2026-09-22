"""Run a named set of benchmarks against a model."""
from __future__ import annotations

import time

import torch

from snapdragon.eval.lambada import evaluate_lambada, load_lambada
from snapdragon.eval.mc import evaluate_mc
from snapdragon.eval.tasks import TASKS
from snapdragon.tokenizer.fast import FastTokenizer

ALL_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "lambada"]


def run_task(
    name: str, model, tok: FastTokenizer, device="cpu", limit: int = 0, max_len: int | None = None,
    autocast=None, **task_kwargs,
) -> dict:
    """Evaluate one task; ``task_kwargs`` (e.g. local file paths) go to the task's loader."""
    t0 = time.time()
    if name == "lambada":
        texts = load_lambada(task_kwargs.get("path"), limit=limit)
        ctx = autocast if autocast is not None else torch.autocast("cpu", enabled=False)
        with ctx:
            result = evaluate_lambada(model, tok, texts, device, max_len)
    elif name in TASKS:
        examples = TASKS.get(name)(limit=limit, **task_kwargs)
        result = evaluate_mc(model, tok, examples, device, max_len, autocast)
    else:
        known = sorted([*TASKS.names(), "lambada"])
        raise KeyError(f"unknown eval task {name!r}; available: {known}")
    result["seconds"] = round(time.time() - t0, 2)
    return result


def run_tasks(
    names: list[str], model, tok: FastTokenizer, device="cpu", limit: int = 0, max_len: int | None = None,
    autocast=None, task_kwargs: dict[str, dict] | None = None,
) -> dict[str, dict]:
    task_kwargs = task_kwargs or {}
    return {
        n: run_task(n, model, tok, device, limit, max_len, autocast, **task_kwargs.get(n, {}))
        for n in names
    }
