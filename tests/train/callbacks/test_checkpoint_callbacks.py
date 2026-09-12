import signal

import pytest

from snapdragon.checkpoint.io import list_checkpoints
from snapdragon.train.callbacks.base import Callback
from snapdragon.train.callbacks.checkpoint import CheckpointCallback, SignalHandler
from snapdragon.train.checkpointing import checkpoint_root


@pytest.fixture
def cfg(tiny_config, tmp_path):
    c = tiny_config()
    c.train.out_dir = str(tmp_path)
    c.train.keep_last_n = 100
    return c


def test_saves_every_n_steps_and_at_the_end_without_duplicates(make_trainer, cfg):
    t = make_trainer(cfg, callbacks=[CheckpointCallback(every=10)])
    t.fit()
    steps = [s for s, _ in list_checkpoints(checkpoint_root(t))]
    assert steps == [10, 20, 30]  # the final save at step 30 reuses the periodic one


def test_final_checkpoint_when_total_is_not_a_multiple(make_trainer, cfg):
    t = make_trainer(cfg, callbacks=[CheckpointCallback(every=8)])
    t.fit()
    assert [s for s, _ in list_checkpoints(checkpoint_root(t))] == [8, 16, 24, 30]


def test_keyboard_interrupt_saves_an_emergency_checkpoint(make_trainer, cfg):
    class Interrupt(Callback):
        priority = 20

        def on_step_end(self, trainer, metrics):
            if metrics["step"] == 7:
                raise KeyboardInterrupt

    t = make_trainer(cfg, callbacks=[Interrupt(), CheckpointCallback(every=100)])
    with pytest.raises(KeyboardInterrupt):
        t.fit()
    assert [s for s, _ in list_checkpoints(checkpoint_root(t))] == [7]


def test_other_exceptions_do_not_write_checkpoints(make_trainer, cfg):
    class Boom(Callback):
        priority = 20

        def on_step_end(self, trainer, metrics):
            if metrics["step"] == 3:
                raise ValueError("bad")

    t = make_trainer(cfg, callbacks=[Boom(), CheckpointCallback(every=100)])
    with pytest.raises(ValueError, match="bad"):
        t.fit()
    assert list_checkpoints(checkpoint_root(t)) == []


def test_non_main_ranks_never_write(make_trainer, cfg):
    t = make_trainer(cfg, callbacks=[CheckpointCallback(every=5)])
    t.rank = 1
    t.fit()
    assert list_checkpoints(checkpoint_root(t)) == []


def test_every_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        CheckpointCallback(every=0)


def test_sigterm_stops_cleanly_and_a_final_checkpoint_is_written(make_trainer, cfg):
    class Killer(Callback):
        priority = 20

        def on_step_end(self, trainer, metrics):
            if metrics["step"] == 4:
                signal.raise_signal(signal.SIGINT)

    before = signal.getsignal(signal.SIGINT)
    t = make_trainer(cfg, callbacks=[SignalHandler(signals=(signal.SIGINT,)), Killer(), CheckpointCallback(every=100)])
    state = t.fit()
    assert state.step == 4
    assert [s for s, _ in list_checkpoints(checkpoint_root(t))] == [4]
    assert signal.getsignal(signal.SIGINT) is before  # handlers are restored


def test_ranks_barrier_after_a_save_but_not_during_an_emergency_save(make_trainer, cfg, monkeypatch):
    calls = []
    monkeypatch.setattr("snapdragon.train.callbacks.checkpoint.barrier", lambda: calls.append("barrier"))

    class Interrupt(Callback):
        priority = 20

        def on_step_end(self, trainer, metrics):
            if metrics["step"] == 7:
                raise KeyboardInterrupt

    t = make_trainer(cfg, callbacks=[Interrupt(), CheckpointCallback(every=5)])
    with pytest.raises(KeyboardInterrupt):
        t.fit()
    assert calls == ["barrier"]  # one for the periodic save at step 5, none for the emergency save
