import pytest

from snapdragon.config.optim import SchedulerConfig
from snapdragon.optim.schedulers import lr_multiplier

N = 1000


def cfg(kind, **kw):
    return SchedulerConfig(kind=kind, warmup_steps=100, decay_fraction=0.2, min_lr_ratio=0.1, **kw)


@pytest.mark.parametrize("kind", ["wsd", "cosine", "linear"])
def test_warmup_is_linear_and_reaches_one(kind):
    c = cfg(kind)
    assert lr_multiplier(c, 0, N) == pytest.approx(0.01)
    assert lr_multiplier(c, 49, N) == pytest.approx(0.5)
    assert lr_multiplier(c, 99, N) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["wsd", "cosine", "linear"])
def test_ends_at_min_ratio(kind):
    c = cfg(kind)
    assert lr_multiplier(c, N, N) == pytest.approx(0.1)
    assert lr_multiplier(c, 10 * N, N) == pytest.approx(0.1)
    assert lr_multiplier(c, N - 1, N) == pytest.approx(0.1, abs=0.03)


def test_wsd_is_flat_then_decays():
    c = cfg("wsd")
    assert lr_multiplier(c, 100, N) == 1.0
    assert lr_multiplier(c, 799, N) == 1.0  # decay begins at step 800
    assert lr_multiplier(c, 800, N) == pytest.approx(1.0)
    assert lr_multiplier(c, 900, N) < 0.6


def test_wsd_resume_anywhere_gives_same_value_regardless_of_history():
    c = cfg("wsd")
    # being a pure function of step is what lets a run be extended or stopped at any checkpoint
    assert lr_multiplier(c, 500, N) == lr_multiplier(c, 500, N)
    assert lr_multiplier(c, 500, 2 * N) == 1.0


@pytest.mark.parametrize("kind", ["cosine", "linear", "wsd"])
def test_monotone_non_increasing_after_warmup(kind):
    c = cfg(kind)
    vals = [lr_multiplier(c, s, N) for s in range(100, N + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_cosine_midpoint():
    c = SchedulerConfig(kind="cosine", warmup_steps=0, min_lr_ratio=0.0)
    assert lr_multiplier(c, 500, 1000) == pytest.approx(0.5)


def test_no_warmup():
    c = SchedulerConfig(kind="linear", warmup_steps=0, min_lr_ratio=0.0)
    assert lr_multiplier(c, 0, 100) == pytest.approx(1.0)
