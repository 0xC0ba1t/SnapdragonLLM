import pytest
import torch

from snapdragon.optim.adamw8bit import AdamW8bit
from snapdragon.optim.hybrid import HybridOptimizer
from snapdragon.optim.muon import Muon


def _setup():
    w = torch.nn.Parameter(torch.randn(8, 8))
    b = torch.nn.Parameter(torch.randn(8))
    return w, b, HybridOptimizer(muon=Muon([w], lr=0.02), adam=AdamW8bit([b], lr=0.01))


def test_step_updates_both_and_zero_grad_clears():
    w, b, opt = _setup()
    w0, b0 = w.detach().clone(), b.detach().clone()
    (w.sum() + b.sum()).backward()
    opt.step()
    assert not torch.equal(w, w0)
    assert not torch.equal(b, b0)
    opt.zero_grad()
    assert w.grad is None
    assert b.grad is None


def test_param_groups_are_the_live_dicts():
    w, b, opt = _setup()
    assert len(opt.param_groups) == 2
    opt.param_groups[0]["lr"] = 123.0
    assert opt.optimizers["muon"].param_groups[0]["lr"] == 123.0


def test_state_dict_roundtrip():
    w, b, opt = _setup()
    (w.sum() + b.sum()).backward()
    opt.step()
    state = opt.state_dict()
    assert set(state) == {"muon", "adam"}
    w2, b2, opt2 = _setup()
    opt2.load_state_dict(state)
    assert "momentum_buffer" in opt2.optimizers["muon"].state[w2]


def test_load_rejects_wrong_layout():
    _, _, opt = _setup()
    with pytest.raises(ValueError, match="mismatch"):
        opt.load_state_dict({"muon": {}})


def test_needs_at_least_one():
    with pytest.raises(ValueError, match="at least one"):
        HybridOptimizer()
