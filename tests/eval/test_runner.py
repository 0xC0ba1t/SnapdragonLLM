import json

import pytest

from snapdragon.config.model import ModelConfig
from snapdragon.eval.runner import ALL_TASKS, run_task, run_tasks
from snapdragon.model.factory import build_model

HS_ROW = {
    "activity_label": "Cooking",
    "ctx_a": "A person is in a kitchen.",
    "ctx_b": "they",
    "endings": ["abc def", "ghi jab", "cde fgh", "ijk abc"],
    "label": "1",
}


@pytest.fixture
def model(small_tokenizer):
    cfg = ModelConfig(vocab_size=small_tokenizer.vocab_size, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
                      mlp_hidden=64, max_seq_len=128)
    return build_model(cfg)


def test_runs_a_task_from_a_local_file(model, small_tokenizer, tmp_path):
    path = tmp_path / "hs.jsonl"
    path.write_text("\n".join(json.dumps(HS_ROW) for _ in range(6)) + "\n")
    out = run_task("hellaswag", model, small_tokenizer, limit=4, max_len=128, path=path)
    assert out["n"] == 4
    assert 0.0 <= out["acc"] <= 1.0
    assert out["seconds"] >= 0


def test_runs_lambada_with_its_own_loader(model, small_tokenizer, tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text(json.dumps({"text": "abc def ghi jab cde"}) + "\n")
    out = run_task("lambada", model, small_tokenizer, path=path, max_len=128)
    assert out["n"] == 1
    assert out["ppl"] > 1


def test_run_tasks_passes_per_task_kwargs(model, small_tokenizer, tmp_path):
    path = tmp_path / "hs.jsonl"
    path.write_text(json.dumps(HS_ROW) + "\n")
    out = run_tasks(["hellaswag"], model, small_tokenizer, max_len=128, task_kwargs={"hellaswag": {"path": path}})
    assert set(out) == {"hellaswag"}
    assert out["hellaswag"]["n"] == 1


def test_unknown_task_lists_the_available_ones(model, small_tokenizer):
    with pytest.raises(KeyError, match="available:.*hellaswag.*lambada"):
        run_task("mmlu", model, small_tokenizer)


def test_default_task_list_is_registered():
    from snapdragon.eval.tasks import TASKS

    for name in ALL_TASKS:
        assert name == "lambada" or name in TASKS
