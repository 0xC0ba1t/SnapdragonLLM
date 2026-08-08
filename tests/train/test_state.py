import pytest

from snapdragon.train.state import TrainState


def test_defaults():
    s = TrainState()
    assert (s.step, s.tokens_seen, s.skipped_steps) == (0, 0, 0)
    assert s.best_val_loss == float("inf")


def test_ema_initialises_with_first_value_then_smooths():
    s = TrainState()
    assert s.update_loss_ema(10.0) == 10.0
    assert s.update_loss_ema(0.0, beta=0.9) == pytest.approx(9.0)


def test_roundtrip():
    s = TrainState(step=5, tokens_seen=100, skipped_steps=1, loss_ema=3.2, best_val_loss=2.9)
    t = TrainState()
    t.load_state_dict(s.state_dict())
    assert t == s


def test_unknown_field_is_rejected():
    with pytest.raises(KeyError, match="bogus"):
        TrainState().load_state_dict({"bogus": 1})
