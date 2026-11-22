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
    """Validation split, from a local jsonl if given, else the Hugging Face copy (same fields as the original).

    The original GitHub repository is no longer available, so there is no stable raw-file URL to fetch."""
    rows = _read_jsonl(path) if path else _hf_rows("Rowan/hellaswag", "default", "validation")
    for i, row in enumerate(rows):
        if limit and i >= limit:
            return
        yield hellaswag_example(row)


def arc_example(row: dict) -> MCExample:
    """AI2 ARC row (HuggingFace ``allenai/ai2_arc`` schema)."""
    labels = list(row["choices"]["label"])
    return MCExample(
        context=f"Question: {row['question']}\nAnswer:",
        choices=[" " + t for t in row["choices"]["text"]],
        label=labels.index(row["answerKey"]),
    )


def _hf_rows(name: str, config: str, split: str) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("install the 'data' extra: pip install -e .[data]") from e
    yield from load_dataset(name, config, split=split)


def _arc(config: str, rows: Iterator[dict] | None, limit: int) -> Iterator[MCExample]:
    for i, row in enumerate(rows if rows is not None else _hf_rows("allenai/ai2_arc", config, "test")):
        if limit and i >= limit:
            return
        yield arc_example(row)


@TASKS.register("arc_easy")
def arc_easy(rows: Iterator[dict] | None = None, limit: int = 0) -> Iterator[MCExample]:
    return _arc("ARC-Easy", rows, limit)


@TASKS.register("arc_challenge")
def arc_challenge(rows: Iterator[dict] | None = None, limit: int = 0) -> Iterator[MCExample]:
    return _arc("ARC-Challenge", rows, limit)


def piqa_example(row: dict, label: int) -> MCExample:
    return MCExample(
        context=f"Question: {row['goal']}\nAnswer:",
        choices=[" " + row["sol1"], " " + row["sol2"]],
        label=label,
    )


@TASKS.register("piqa")
def piqa(path: str | Path | None = None, labels_path: str | Path | None = None, limit: int = 0) -> Iterator[MCExample]:
    """PIQA validation set: questions in jsonl, answers in a separate one-label-per-line file."""
    rows = _read_jsonl(path or fetch(PIQA_URL))
    labels = Path(labels_path or fetch(PIQA_LABELS_URL)).read_text().split()
    for i, (row, label) in enumerate(zip(rows, labels, strict=False)):
        if limit and i >= limit:
            return
        yield piqa_example(row, int(label))
