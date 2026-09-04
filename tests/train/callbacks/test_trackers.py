import sys
import types

import pytest

from snapdragon.config.schema import Config
from snapdragon.train.callbacks.trackers import TensorBoardLogger, WandbLogger, _scalars


class FakeTrainer:
    is_main = True
    cfg = Config()


def test_scalars_drop_non_numeric_and_bookkeeping_fields():
    out = _scalars({"step": 3, "loss": 2, "lr": 1e-3, "skipped": False, "note": "x", "flag": True})
    assert out == {"loss": 2.0, "lr": 1e-3}


def test_tensorboard_writes_scalars_every_n_steps(tmp_path, monkeypatch):
    calls = []

    class FakeWriter:
        def __init__(self, log_dir):
            calls.append(("open", log_dir))

        def add_scalar(self, tag, value, step):
            calls.append((tag, value, step))

        def close(self):
            calls.append("close")

    fake_mod = types.ModuleType("torch.utils.tensorboard")
    fake_mod.SummaryWriter = FakeWriter
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_mod)

    cb = TensorBoardLogger(tmp_path, every=2)
    cb.on_train_start(FakeTrainer())
    cb.on_step_end(FakeTrainer(), {"step": 1, "loss": 3.0})
    cb.on_step_end(FakeTrainer(), {"step": 2, "loss": 2.5, "skipped": False})
    cb.on_train_end(FakeTrainer())
    assert calls == [("open", str(tmp_path)), ("train/loss", 2.5, 2), "close"]


def test_missing_tensorboard_gives_an_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)  # makes the import raise ImportError
    with pytest.raises(ImportError, match="pip install"):
        TensorBoardLogger(tmp_path).on_train_start(FakeTrainer())


def test_wandb_logs_and_finishes(monkeypatch):
    log = []

    class Run:
        def log(self, data, step):
            log.append((data, step))

        def finish(self):
            log.append("finish")

    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: (log.append(("init", kw["project"], kw["name"], kw["mode"])) or Run())
    monkeypatch.setitem(sys.modules, "wandb", fake)

    cb = WandbLogger("proj", every=5, mode="offline")
    cb.on_train_start(FakeTrainer())
    cb.on_step_end(FakeTrainer(), {"step": 5, "loss": 1.5, "tokens_per_sec": 100})
    cb.on_step_end(FakeTrainer(), {"step": 6, "loss": 1.4})
    cb.on_train_end(FakeTrainer())
    assert log[0] == ("init", "proj", "snapdragon-500m", "offline")
    assert log[1] == ({"loss": 1.5, "tokens_per_sec": 100.0}, 5)
    assert log[-1] == "finish"
    assert len(log) == 3


def test_non_main_rank_does_nothing(tmp_path):
    class Worker(FakeTrainer):
        is_main = False

    for cb in (TensorBoardLogger(tmp_path), WandbLogger("p")):
        cb.on_train_start(Worker())
        cb.on_step_end(Worker(), {"step": 10, "loss": 1.0})
        cb.on_train_end(Worker())
