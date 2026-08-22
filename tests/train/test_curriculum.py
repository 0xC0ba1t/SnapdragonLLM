import pytest

from snapdragon.config.train import TrainConfig
from snapdragon.train.curriculum import divisors, seq_len_at


def cfg(**kw):
    base = dict(seq_len=2048, seq_len_start=256, seq_len_warmup_tokens=1_000_000)
    base.update(kw)
    return TrainConfig(**base)


def test_divisors():
    assert divisors(12) == [1, 2, 3, 4, 6, 12]


def test_disabled_returns_full_length():
    assert seq_len_at(0, cfg(seq_len_warmup_tokens=0)) == 2048


def test_starts_at_start_and_ends_at_full_length():
    c = cfg()
    assert seq_len_at(0, c) == 256
    assert seq_len_at(1_000_000, c) == 2048
    assert seq_len_at(10_000_000, c) == 2048


def test_grows_monotonically_through_powers_of_two():
    c = cfg()
    seen = [seq_len_at(t, c) for t in range(0, 1_000_001, 10_000)]
    assert seen == sorted(seen)
    assert set(seen) == {256, 512, 1024, 2048}


def test_always_divides_the_full_length():
    c = cfg(seq_len=1536, seq_len_start=96, global_batch_tokens=1536 * 4)
    for t in range(0, 1_000_001, 25_000):
        assert 1536 % seq_len_at(t, c) == 0


def test_start_equal_to_full_is_constant():
    assert {seq_len_at(t, cfg(seq_len_start=2048)) for t in (0, 5, 999_999)} == {2048}


def test_invalid_start_is_rejected_by_config():
    with pytest.raises(ValueError, match="seq_len_start"):
        cfg(seq_len_start=4096)
