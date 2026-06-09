import pytest

from snapdragon.config.train import TrainConfig


def test_grad_accum_default():
    assert TrainConfig().grad_accum_steps() == 64  # 524288 / (4 * 2048)


def test_grad_accum_scales_with_world_size():
    assert TrainConfig().grad_accum_steps(world_size=8) == 8


def test_indivisible_batch_rejected():
    with pytest.raises(ValueError, match="not divisible"):
        TrainConfig(global_batch_tokens=1000).grad_accum_steps()


def test_total_steps():
    assert TrainConfig(max_tokens=524_288 * 10).total_steps == 10


def test_bad_precision():
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="fp8")


def test_seq_len_warmup_bounds():
    with pytest.raises(ValueError, match="seq_len_start"):
        TrainConfig(seq_len_warmup_tokens=100, seq_len_start=4096, seq_len=2048)
