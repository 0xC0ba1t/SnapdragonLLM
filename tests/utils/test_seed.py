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


class _OnGpu:
    """Stands in for a tensor that a cuda `map_location` moved off the CPU."""

    def __init__(self, t):
        self.t = t

    def cpu(self):
        return self.t


def test_restore_moves_states_back_to_cpu_first():
    seed_everything(3)
    state = get_rng_state()
    a = torch.randn(3)
    state["torch"] = _OnGpu(state["torch"])  # torch.set_rng_state rejects anything but a CPU ByteTensor
    set_rng_state(state)
    assert torch.equal(a, torch.randn(3))


def test_restore_ignores_cuda_states_beyond_the_visible_devices(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda s, i: seen.append((s, i)))
    state = get_rng_state()
    state.pop("cuda", None)
    state["cuda"] = [_OnGpu("s0"), _OnGpu("s1")]
    set_rng_state(state)
    assert seen == [("s0", 0)]
