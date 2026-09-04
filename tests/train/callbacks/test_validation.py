import math

import pytest
import torch

from snapdragon.data.loader import Batch
from snapdragon.train.callbacks.validation import ValLossCallback, evaluate_loss


def _val_batches(n=3, seed=5):
    """Same cyclic structure as the training fixture, so a trained model scores better."""
    pattern = (torch.arange(2048) * 5 + 3 + seed) % 64
    out = []
    for i in range(n):
        w = torch.stack([pattern[(i * 2 + j) * 11 : (i * 2 + j) * 11 + 17] for j in range(2)])
        out.append(Batch(w[:, :-1].contiguous(), w[:, 1:].contiguous()))
    return out


def test_untrained_val_loss_is_near_log_vocab(make_trainer):
    t = make_trainer()
    loss = evaluate_loss(t, _val_batches())
    assert abs(loss - math.log(64)) < 0.6


def test_training_lowers_val_loss(make_trainer):
    t = make_trainer()
    before = evaluate_loss(t, _val_batches(seed=0))
    for _ in range(25):
        t.train_step()
    assert evaluate_loss(t, _val_batches(seed=0)) < before - 0.5


def test_evaluate_restores_train_mode_and_does_not_touch_grads(make_trainer):
    t = make_trainer()
    t.model.train()
    evaluate_loss(t, _val_batches(1))
    assert t.model.training
    assert all(p.grad is None for p in t.raw_model.parameters())


def test_callback_adds_metrics_on_schedule_and_tracks_best(make_trainer):
    cb = ValLossCallback(_val_batches(2), every=5)
    t = make_trainer(callbacks=[cb])
    m3, m5 = {"step": 3}, {"step": 5}
    cb.on_step_end(t, m3)
    cb.on_step_end(t, m5)
    assert "val_loss" not in m3
    assert m5["val_ppl"] == pytest.approx(math.exp(m5["val_loss"]))
    assert t.state.best_val_loss == m5["val_loss"]


def test_always_evaluates_on_the_final_step(make_trainer):
    cb = ValLossCallback(_val_batches(1), every=1000)
    t = make_trainer(callbacks=[cb])
    m = {"step": t.total_steps}
    cb.on_step_end(t, m)
    assert "val_loss" in m


def test_runs_inside_fit_before_loggers(make_trainer):
    seen = []

    from snapdragon.train.callbacks.base import Callback

    class Spy(Callback):
        priority = 50

        def on_step_end(self, trainer, metrics):
            if "val_loss" in metrics:
                seen.append(metrics["step"])

    t = make_trainer(callbacks=[Spy(), ValLossCallback(_val_batches(1), every=10)])
    t.fit()
    assert seen == [10, 20, 30]
