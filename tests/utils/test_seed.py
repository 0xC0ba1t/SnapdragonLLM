import torch

from snapdragon.utils.seed import get_rng_state, seed_everything, set_rng_state


def test_seed_is_reproducible():
    seed_everything(7)
    a = torch.randn(4)
    seed_everything(7)
    assert torch.equal(a, torch.randn(4))


def test_ranks_get_different_streams():
    seed_everything(7, rank=0)
    a = torch.randn(4)
    seed_everything(7, rank=1)
    assert not torch.equal(a, torch.randn(4))


def test_rng_state_roundtrip():
    seed_everything(1)
    state = get_rng_state()
    a = torch.randn(3)
    set_rng_state(state)
    assert torch.equal(a, torch.randn(3))
