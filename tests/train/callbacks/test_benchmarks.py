import json

import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.model.factory import build_model
from snapdragon.train.callbacks.benchmarks import BenchmarkCallback

ROW = {"activity_label": "Cooking", "ctx_a": "A person is in a kitchen.", "ctx_b": "they",
       "endings": ["abc def", "ghi jab", "cde fgh", "ijk abc"], "label": "0"}


class FakeTrainer:
    is_main = True
    total_steps = 100
    device = torch.device("cpu")

    class cfg:  # noqa: N801
        class train:  # noqa: N801
            precision = "fp32"

    def __init__(self, model):
        self.raw_model = model


@pytest.fixture
def setup(small_tokenizer, tmp_path):
    cfg = ModelConfig(vocab_size=small_tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                      mlp_hidden=32, max_seq_len=64)
    path = tmp_path / "hs.jsonl"
    path.write_text("\n".join(json.dumps(ROW) for _ in range(3)) + "\n")
    cb = BenchmarkCallback(small_tokenizer, ["hellaswag"], every=10, limit=2, task_kwargs={"hellaswag": {"path": path}})
    return cb, FakeTrainer(build_model(cfg))


def test_adds_task_metrics_on_schedule(setup):
    cb, tr = setup
    skipped, hit = {"step": 5}, {"step": 10}
    cb.on_step_end(tr, skipped)
    cb.on_step_end(tr, hit)
    assert "hellaswag_acc" not in skipped
    assert 0.0 <= hit["hellaswag_acc"] <= 1.0
    assert 0.0 <= hit["hellaswag_acc_norm"] <= 1.0


def test_evaluates_on_the_final_step(setup):
    cb, tr = setup
    m = {"step": 100}
    cb.on_step_end(tr, m)
    assert "hellaswag_acc" in m


def test_non_main_ranks_skip(setup):
    cb, tr = setup
    tr.is_main = False
    m = {"step": 10}
    cb.on_step_end(tr, m)
    assert "hellaswag_acc" not in m


def test_model_is_returned_to_train_mode(setup):
    cb, tr = setup
    tr.raw_model.train()
    cb.on_step_end(tr, {"step": 10})
    assert tr.raw_model.training
