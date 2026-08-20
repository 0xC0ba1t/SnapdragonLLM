import copy

import pytest
import torch

from snapdragon.checkpoint.io import list_checkpoints, load_checkpoint
from snapdragon.train.checkpointing import (
    checkpoint_root,
    maybe_resume,
    resolve_resume_path,
    restore_trainer,
    save_trainer_checkpoint,
    trainer_components,
)


@pytest.fixture
def cfg(tiny_config, tmp_path):
    c = tiny_config()
    c.train.out_dir = str(tmp_path)
    return c


def test_components_cover_everything_needed_to_resume(make_trainer, cfg):
    t = make_trainer(cfg)
    t.train_step()
    comps = trainer_components(t)
    assert set(comps) == {"model", "optim", "schedule", "loader", "trainer_state", "rng", "meta"}
    assert comps["trainer_state"]["step"] == 1
    assert comps["loader"] == {"batch_index": 2}  # two micro-batches were consumed
    assert comps["meta"]["config"]["model"]["n_layers"] == 2


def test_save_creates_checkpoint_and_prunes(make_trainer, cfg):
    cfg.train.keep_last_n = 2
    t = make_trainer(cfg)
    for _ in range(4):
        t.train_step()
        save_trainer_checkpoint(t)
    steps = [s for s, _ in list_checkpoints(checkpoint_root(t))]
    assert steps == [3, 4]


def test_resumed_trainer_continues_exactly_like_the_original(make_trainer, cfg):
    """Train 4 steps, checkpoint, train 3 more. A fresh trainer restored from the checkpoint and
    run for 3 steps must land on bit-identical weights."""
    a = make_trainer(cfg)
    for _ in range(4):
        a.train_step()
    path = save_trainer_checkpoint(a)
    for _ in range(3):
        a.train_step()

    other = copy.deepcopy(cfg)
    other.model.init_seed = 999  # different init: everything must come from the checkpoint
    b = make_trainer(other)
    assert not torch.equal(a.raw_model.transformer.embed.weight, b.raw_model.transformer.embed.weight)
    restore_trainer(b, load_checkpoint(path))
    assert b.state.step == 4
    for _ in range(3):
        b.train_step()

    for (n, pa), (_, pb) in zip(a.raw_model.named_parameters(), b.raw_model.named_parameters()):
        assert torch.equal(pa, pb), n
    assert a.state.state_dict() == b.state.state_dict()


def test_resolve_resume_modes(make_trainer, cfg):
    t = make_trainer(cfg)
    assert resolve_resume_path(t, "none") is None
    assert resolve_resume_path(t, "auto") is None  # nothing saved yet
    t.train_step()
    saved = save_trainer_checkpoint(t)
    assert resolve_resume_path(t, "auto") == saved
    assert str(resolve_resume_path(t, "some/explicit/path")).endswith("path")


def test_maybe_resume_respects_config(make_trainer, cfg):
    t = make_trainer(cfg)
    t.train_step()
    save_trainer_checkpoint(t)

    cfg.train.resume = "none"
    assert maybe_resume(make_trainer(cfg)) is False
    cfg.train.resume = "auto"
    fresh = make_trainer(cfg, seed=9)
    assert maybe_resume(fresh) is True
    assert fresh.state.step == 1
