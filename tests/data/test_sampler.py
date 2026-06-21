import numpy as np
import pytest

from snapdragon.data.sampler import EpochPermutation, Sampler


def test_epoch_permutation_covers_every_item_once_per_epoch():
    p = EpochPermutation(10, seed=1, stream=0)
    first = [p.index(c) for c in range(10)]
    second = [p.index(c) for c in range(10, 20)]
    assert sorted(first) == list(range(10))
    assert sorted(second) == list(range(10))
    assert first != second  # reshuffled


def test_epoch_permutation_is_reproducible_and_random_access():
    a, b = EpochPermutation(7, 5, 2), EpochPermutation(7, 5, 2)
    assert [a.index(c) for c in range(21)] == [b.index(c) for c in reversed(range(21))][::-1]


def test_empty_source_rejected():
    with pytest.raises(ValueError, match="at least one"):
        EpochPermutation(0, 0, 0)


def test_sampler_sequential_equals_random_access():
    s1 = Sampler([30, 50], [1, 2], seed=9)
    seq = [s1.locate(g) for g in range(300)]
    s2 = Sampler([30, 50], [1, 2], seed=9)
    rev = [s2.locate(g) for g in reversed(range(300))][::-1]
    assert seq == rev


def test_no_window_repeats_within_an_epoch_of_a_source():
    s = Sampler([20], [1], seed=0)
    wins = [s.locate(g)[1] for g in range(20)]
    assert sorted(wins) == list(range(20))


def test_each_source_sees_its_own_windows():
    s = Sampler([5, 1000], [1, 1], seed=3)
    for g in range(200):
        src, w = s.locate(g)
        assert 0 <= w < [5, 1000][src]


def test_seek_over_block_boundaries():
    s = Sampler([100, 100], [1, 1], seed=2)
    a = s.locate(9000)
    s2 = Sampler([100, 100], [1, 1], seed=2)
    for g in range(0, 9001, 250):
        s2.locate(g)
    assert s2.locate(9000) == a
    assert np.all(s.mixture.counts_before(9000) == s._counts)
