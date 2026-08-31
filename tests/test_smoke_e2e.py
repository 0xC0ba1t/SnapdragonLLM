"""The whole pipeline on a toy corpus, on CPU: tokenizer -> shards -> train -> checkpoint -> resume."""
import importlib.util
import json
import random
from pathlib import Path

import pytest

from snapdragon.checkpoint.io import list_checkpoints
from snapdragon.data.tokenize_shards import tokenize_to_shards
from snapdragon.tokenizer.fast import FastTokenizer, train_fast_tokenizer

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    rng = random.Random(0)
    vocab_words = ["".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(2, 6))) for _ in range(120)]
    docs = [" ".join(rng.choice(vocab_words) for _ in range(rng.randint(40, 120))) for _ in range(1200)]
    tok = FastTokenizer(train_fast_tokenizer(docs, 512))
    tok.save(tmp / "tok.json")
    tokenize_to_shards(docs, tok, tmp / "data", shard_tokens=60_000, val_tokens=4_000)
    return tmp


def _overrides(tmp, name, extra=()):
    return [
        f"data.sources=[{{name: toy, path: {(tmp / 'data').as_posix()}}}]",
        f"train.out_dir={(tmp / 'runs').as_posix()}",
        f"train.run_name={name}",
        "train.max_tokens=61440",  # 30 steps of 2048 tokens
        "train.save_every=15",
        "train.log_every=5",
        *extra,
    ]


def _losses(tmp, name):
    rows = [json.loads(line) for line in (tmp / "runs" / name / "metrics.jsonl").read_text().splitlines()]
    return [r["loss"] for r in rows]


def test_train_cli_learns_checkpoints_and_resumes(prepared):
    train = _load_script("train")
    cfg = str(ROOT / "configs" / "train" / "debug-cpu.yaml")
    train.main(["--config", cfg, "--device", "cpu", *_overrides(prepared, "e2e")])

    losses = _losses(prepared, "e2e")
    assert len(losses) == 6  # logged every 5 steps of 30
    assert losses[-1] < losses[0] - 0.5
    ckpts = [s for s, _ in list_checkpoints(prepared / "runs" / "e2e" / "checkpoints")]
    assert ckpts == [15, 30]

    # extend the run: it must pick up from step 30 rather than start over
    train.main(["--config", cfg, "--device", "cpu", *_overrides(prepared, "e2e", ["train.max_tokens=81920"])])
    steps = [json.loads(line)["step"] for line in (prepared / "runs" / "e2e" / "metrics.jsonl").read_text().splitlines()]
    assert steps[-1] == 40
    assert steps.count(30) == 1  # not retrained
