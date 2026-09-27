import importlib.util
import json
from pathlib import Path

import pytest

from snapdragon.train.checkpointing import checkpoint_root, save_trainer_checkpoint

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval.py"


def _cli():
    spec = importlib.util.spec_from_file_location("eval_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ckpt(make_trainer, tiny_config, tmp_path, small_tokenizer):
    cfg = tiny_config()
    cfg.model.vocab_size = small_tokenizer.vocab_size  # so the tokenizer's ids are valid for the model
    cfg.train.out_dir = str(tmp_path)
    t = make_trainer(cfg)
    t.train_step()
    save_trainer_checkpoint(t)
    return checkpoint_root(t)


def test_perplexity_from_a_prepared_dataset(ckpt, tmp_path, make_token_dataset):
    # the tiny model has vocab 64; build a dataset whose ids fit and which has a val split
    from snapdragon.data.manifest import DatasetManifest, ShardInfo
    from snapdragon.data.shards import ShardWriter

    data = tmp_path / "ds"
    with ShardWriter(data, "val", 10**9, eos_id=63) as w:
        w.add_document([(i * 5 + 3) % 63 for i in range(2000)])
    m = DatasetManifest(vocab_size=64, eos_id=63, shards=[ShardInfo(s, t, d, "val") for s, t, d in w.files])
    m.save(data)
    out = tmp_path / "res.json"
    res = _cli().main(["--checkpoint", str(ckpt), "--perplexity", str(data), "--max-batches", "3",
                       "--precision", "fp32", "--device", "cpu", "--out", str(out)])
    assert res["perplexity"]["tokens"] == 3 * 4 * 16
    assert json.loads(out.read_text())["perplexity"]["ppl"] > 1


def test_tasks_require_a_tokenizer(ckpt):
    with pytest.raises(SystemExit):
        _cli().main(["--checkpoint", str(ckpt), "--tasks", "hellaswag"])


def test_runs_a_task_with_a_tokenizer(ckpt, small_tokenizer, tmp_path, monkeypatch):
    small_tokenizer.save(tmp_path / "tok.json")
    from snapdragon.eval.mc import MCExample

    monkeypatch.setitem(
        __import__("snapdragon.eval.tasks", fromlist=["TASKS"]).TASKS._items,
        "hellaswag",
        lambda limit=0, **kw: iter([MCExample("abc", ["def", "ghi"], 0)] * 2),
    )
    res = _cli().main(["--checkpoint", str(ckpt), "--tokenizer", str(tmp_path / "tok.json"),
                       "--tasks", "hellaswag", "--device", "cpu", "--precision", "fp32"])
    assert res["hellaswag"]["n"] == 2
