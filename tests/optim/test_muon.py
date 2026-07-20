import pytest
import torch

from snapdragon.optim.muon import Muon


def _problem(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(16, 24, generator=g) * 0.1)
    target = torch.randn(16, 24, generator=g)
    return w, target


def test_reduces_a_simple_loss():
    w, target = _problem()
    opt = Muon([w], lr=0.02, weight_decay=0.0)
    first = ((w - target) ** 2).mean().item()
    for _ in range(100):
        opt.zero_grad()
        ((w - target) ** 2).mean().backward()
        opt.step()
    assert ((w - target) ** 2).mean().item() < 0.5 * first


def test_update_has_flat_spectrum():
    """The step taken in one iteration is (nearly) orthogonal: singular values are all alike."""
    w, target = _problem(1)
    start = w.detach().clone()
    opt = Muon([w], lr=1.0, weight_decay=0.0, momentum=0.0, nesterov=False)
    ((w - target) ** 2).mean().backward()
    opt.step()
    step = start - w.detach()
    s = torch.linalg.svdvals(step)
    assert s.min() > 0.3 * s.max()


def test_weight_decay_shrinks_weights_when_gradient_is_zero():
    w = torch.nn.Parameter(torch.ones(4, 4))
    w.grad = torch.zeros(4, 4)
    Muon([w], lr=0.1, weight_decay=0.5, momentum=0.0).step()
    assert torch.allclose(w, torch.full((4, 4), 1 - 0.1 * 0.5), atol=1e-3)


def test_momentum_buffer_is_bf16_and_persists():
    w, target = _problem()
    opt = Muon([w])
    ((w - target) ** 2).mean().backward()
    opt.step()
    buf = opt.state[w]["momentum_buffer"]
    assert buf.dtype == torch.bfloat16
    assert buf.abs().sum() > 0


def test_rejects_non_matrices_and_bad_args():
    with pytest.raises(ValueError, match="2-D"):
        Muon([torch.nn.Parameter(torch.zeros(3))])
    with pytest.raises(ValueError, match="lr"):
        Muon([torch.nn.Parameter(torch.zeros(3, 3))], lr=0)
    with pytest.raises(ValueError, match="momentum"):
        Muon([torch.nn.Parameter(torch.zeros(3, 3))], momentum=1.0)


def test_state_dict_roundtrip_continues_identically():
    w1, target = _problem()
    w2 = torch.nn.Parameter(w1.detach().clone())
    o1, o2 = Muon([w1]), Muon([w2])

    def run(w, opt, n):
        for _ in range(n):
            opt.zero_grad()
            ((w - target) ** 2).mean().backward()
            opt.step()

    run(w1, o1, 5)
    w2.data.copy_(w1.data)
    o2.load_state_dict(o1.state_dict())
    run(w1, o1, 3)
    run(w2, o2, 3)
    assert torch.allclose(w1, w2, atol=1e-6)


def test_resume_keeps_buffers_in_bf16():
    w, target = _problem()
    opt = Muon([w])
    ((w - target) ** 2).mean().backward()
    opt.step()
    fresh = Muon([w])
    fresh.load_state_dict(opt.state_dict())
    assert fresh.state[w]["momentum_buffer"].dtype == torch.bfloat16
