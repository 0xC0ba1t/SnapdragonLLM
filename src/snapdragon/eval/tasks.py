"""Benchmark definitions: each task turns a data file into MCExamples (or LAMBADA's own format).

Preprocessing follows lm-evaluation-harness so numbers are comparable with published ones.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

from snapdragon.eval.data import fetch
from snapdragon.eval.mc import MCExample
from snapdragon.utils.registry import Registry

TASKS: Registry[Callable[..., Iterator[MCExample]]] = Registry("eval task")

HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
PIQA_URL = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl"
PIQA_LABELS_URL = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst"


def _read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def hellaswag_preprocess(text: str) -> str:
    """Clean wikiHow artefacts: ' [title]' -> '. ', drop bracketed tags, collapse double spaces."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag_example(row: dict) -> MCExample:
    ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
    return MCExample(
        context=hellaswag_preprocess(row["activity_label"] + ": " + ctx),
        choices=[" " + hellaswag_preprocess(e) for e in row["endings"]],
        label=int(row["label"]),
    )


@TASKS.register("hellaswag")
def hellaswag(path: str | Path | None = None, limit: int = 0) -> Iterator[MCExample]:
    for i, row in enumerate(_read_jsonl(path or fetch(HELLASWAG_URL))):
        if limit and i >= limit:
            return
        yield hellaswag_example(row)
