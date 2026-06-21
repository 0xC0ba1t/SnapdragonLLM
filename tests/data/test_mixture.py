import numpy as np
import pytest

from snapdragon.data.mixture import BLOCK, Mixture


def test_deterministic_for_a_seed():
    a, b = Mixture([1, 1], seed=3), Mixture([1, 1], seed=3)
    assert [a.source_of(i) for i in range(50)] == [b.source_of(i) for i in range(50)]


def test_different_seeds_differ():
    a, b = Mixture([1, 1], seed=3), Mixture([1, 1], seed=4)
    assert [a.source_of(i) for i in range(200)] != [b.source_of(i) for i in range(200)]


def test_proportions_follow_weights():
    m = Mixture([3, 1], seed=0)
    counts = m.counts_before(20 * BLOCK)
    assert counts.sum() == 20 * BLOCK
    assert counts[0] / counts.sum() == pytest.approx(0.75, abs=0.01)


def test_counts_before_matches_brute_force_across_block_boundary():
    m = Mixture([2, 1, 1], seed=1)
    g = BLOCK + 123
    brute = np.bincount([m.source_of(i) for i in range(g)], minlength=3)
    assert m.counts_before(g).tolist() == brute.tolist()


def test_bad_weights():
    with pytest.raises(ValueError, match="positive"):
        Mixture([1, 0], seed=0)
    with pytest.raises(ValueError, match="positive"):
        Mixture([], seed=0)
