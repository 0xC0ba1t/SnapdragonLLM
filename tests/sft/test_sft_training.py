import json

import pytest
import torch

from snapdragon.data.packing import IGNORE_INDEX
from snapdragon.train.setup import build_trainer


@pytest.fixture
def sft_cfg(tiny_config, small_tokenizer, tmp_path):
    small_tokenizer.save(tmp_path / "tok.json")
    # a learnable mapping: the answer is a fixed function of the question
    rows = []
    for i in range(24):
        q = ["abc", "def", "ghi", "jab"][i % 4]
        a = {"abc": "def def", "def": "ghi ghi", "ghi": "jab jab", "jab": "abc abc"}[q]
        rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    (tmp_path / "sft.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg = tiny_config()
    cfg.model.vocab_size = small_tokenizer.vocab_size
    cfg.model.max_seq_len = cfg.train.seq_len = cfg.data.seq_len = 32
    cfg.train.task = "sft"
    cfg.train.global_batch_tokens = 2 * 32 * 2
    cfg.train.max_tokens = cfg.train.global_batch_tokens * 40
    cfg.train.out_dir = str(tmp_path / "runs")
    cfg.train.compile = "false"
    cfg.train.log_every = 1
    cfg.train.save_every = 1000
    cfg.data.sft_path = str(tmp_path / "sft.jsonl")
    cfg.data.tokenizer_path = str(tmp_path / "tok.json")
    cfg.optim.adam_lr = 5e-3
    return cfg


def test_sft_run_learns_the_answers_and_only_trains_on_assistant_tokens(sft_cfg):
    trainer = build_trainer(sft_cfg, device="cpu")
    batch = next(trainer.train_loader)
    assert (batch.y == IGNORE_INDEX).any()  # user tokens and padding are masked out
    assert (batch.y != IGNORE_INDEX).any()
    trainer.train_loader.batch_index = 0

    losses = [trainer.train_step()["loss"] for _ in range(40)]
    assert losses[-1] < 0.5 * losses[0]
    assert all(torch.isfinite(torch.tensor(losses)))


def test_sft_without_a_dataset_path_fails_early(sft_cfg):
    sft_cfg.data.sft_path = ""
    with pytest.raises(ValueError, match="sft_path"):
        sft_cfg.validate()
