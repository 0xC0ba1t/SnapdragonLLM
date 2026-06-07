import pytest

from snapdragon.utils.timer import Timer, timed


def test_timer_accumulates():
    t = Timer()
    for _ in range(3):
        with t:
            pass
    assert t.count == 3
    assert t.total >= 0.0
    assert t.mean == pytest.approx(t.total / 3)


def test_stop_without_start_raises():
    with pytest.raises(RuntimeError, match="before start"):
        Timer().stop()


def test_timed_context_records_seconds():
    with timed() as t:
        pass
    assert t["seconds"] >= 0.0
