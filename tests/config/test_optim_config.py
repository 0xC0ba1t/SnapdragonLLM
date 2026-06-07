import pytest

from snapdragon.config.optim import OptimConfig, SchedulerConfig


def test_defaults_construct():
    cfg = OptimConfig()
    assert cfg.scheduler.kind == "wsd"
    assert cfg.eight_bit


def test_bad_scheduler_kind():
    with pytest.raises(ValueError, match="scheduler kind"):
        SchedulerConfig(kind="exponential")


def test_bad_fractions():
    with pytest.raises(ValueError, match="decay_fraction"):
        SchedulerConfig(decay_fraction=1.5)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        SchedulerConfig(min_lr_ratio=-0.1)


def test_negative_lr_rejected():
    with pytest.raises(ValueError, match="positive"):
        OptimConfig(adam_lr=0.0)
