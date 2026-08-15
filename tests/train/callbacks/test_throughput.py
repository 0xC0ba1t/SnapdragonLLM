import pytest

from snapdragon.config.model import ModelConfig
from snapdragon.train.callbacks.throughput import ThroughputCallback


class _Dev:
    type = "cpu"


class _Model:
    cfg = ModelConfig()


class _Cfg:
    class train:  # noqa: N801
        seq_len = 2048


class _State:
    tokens_seen = 0


class FakeTrainer:
    device = _Dev()
    world_size = 1
    raw_model = _Model()
    cfg = _Cfg()

    def __init__(self):
        self.state = _State()


def test_first_step_has_no_throughput_then_rates_appear(monkeypatch):
    clock = iter([100.0, 102.0, 103.0])
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    tr = FakeTrainer()
    cb = ThroughputCallback(peak_tflops=100.0)
    cb.on_train_start(tr)

    tr.state.tokens_seen = 1000
    m1 = {}
    cb.on_step_end(tr, m1)
    assert "tokens_per_sec" not in m1

    tr.state.tokens_seen = 5000  # +4000 tokens in 2.0 s
    m2 = {}
    cb.on_step_end(tr, m2)
    assert m2["step_time"] == pytest.approx(2.0)
    assert m2["tokens_per_sec"] == pytest.approx(2000.0)
    assert 0 < m2["mfu"] < 1

    tr.state.tokens_seen = 7000  # +2000 in 1.0 s
    m3 = {}
    cb.on_step_end(tr, m3)
    assert m3["tokens_per_sec"] == pytest.approx(2000.0)


def test_no_mfu_without_a_known_peak(monkeypatch):
    clock = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    tr = FakeTrainer()
    cb = ThroughputCallback()
    cb.on_train_start(tr)
    for tokens in (10, 20):
        tr.state.tokens_seen = tokens
        m = {}
        cb.on_step_end(tr, m)
    assert "mfu" not in m
    assert "tokens_per_sec" in m


def test_runs_before_loggers():
    assert ThroughputCallback.priority < 50
