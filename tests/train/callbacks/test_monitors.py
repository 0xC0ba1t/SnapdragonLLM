import pytest
import torch

from snapdragon.train.callbacks.monitors import (
    DivergenceError,
    DivergenceGuard,
    MemoryCallback,
    WeightNormCallback,
)


class Dev:
    type = "cpu"


class Raw:
    def __init__(self):
        self.p = [torch.nn.Parameter(torch.full((4,), 3.0)), torch.nn.Parameter(torch.full((9,), 4.0))]

    def parameters(self):
        return iter(self.p)


class FakeTrainer:
    device = Dev()

    def __init__(self):
        self.raw_model = Raw()


def test_memory_callback_is_a_noop_on_cpu():
    cb = MemoryCallback()
    m = {"step": 1}
    cb.on_train_start(FakeTrainer())
    cb.on_step_end(FakeTrainer(), m)
    assert m == {"step": 1}


def test_weight_norm_is_global_l2_and_respects_every():
    cb = WeightNormCallback(every=2)
    tr = FakeTrainer()
    m1, m2 = {"step": 1}, {"step": 2}
    cb.on_step_end(tr, m1)
    cb.on_step_end(tr, m2)
    assert "param_norm" not in m1
    expected = (4 * 9.0 + 9 * 16.0) ** 0.5  # sqrt(sum of squares)
    assert m2["param_norm"] == pytest.approx(expected, rel=1e-5)


def test_guard_resets_on_a_good_step_and_raises_on_a_streak():
    g = DivergenceGuard(max_consecutive_skips=3)
    for skipped in (True, True, False, True, True):
        g.on_step_end(None, {"step": 1, "loss": 1.0, "skipped": skipped})
    with pytest.raises(DivergenceError, match="3 consecutive skipped steps at step 9"):
        g.on_step_end(None, {"step": 9, "loss": float("nan"), "skipped": True})


def test_guard_ignores_metrics_without_skipped_key():
    g = DivergenceGuard(max_consecutive_skips=1)
    g.on_step_end(None, {"step": 1, "loss": 1.0})
    assert g.streak == 0
