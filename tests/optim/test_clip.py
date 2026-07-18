import pytest
import torch

from snapdragon.optim.clip import clip_grad_norm


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(5, 3, generator=g)), torch.nn.Parameter(torch.randn(7, generator=g))]
    for p in ps:
        p.grad = torch.randn_like(p, generator=g) * 10
    return ps


def test_matches_torch_clip_grad_norm():
    a, b = _params(), _params()
    ref_norm = torch.nn.utils.clip_grad_norm_(a, 1.0)
    norm = clip_grad_norm(b, 1.0)
    assert torch.allclose(norm, ref_norm, atol=1e-5)
    for p, q in zip(a, b):
        assert torch.allclose(p.grad, q.grad, atol=1e-5)


def test_small_gradients_are_untouched():
    ps = _params()
    before = [p.grad.clone() for p in ps]
    clip_grad_norm(ps, 1e6)
    for p, g in zip(ps, before):
        assert torch.allclose(p.grad, g)


def test_zero_disables_clipping_but_still_reports_norm():
    ps = _params()
    before = [p.grad.clone() for p in ps]
    norm = clip_grad_norm(ps, 0.0)
    assert norm.item() > 1
    for p, g in zip(ps, before):
        assert torch.equal(p.grad, g)


def test_non_finite_norm_leaves_grads_alone_and_is_detectable():
    ps = _params()
    ps[0].grad[0, 0] = float("inf")
    before = ps[1].grad.clone()
    norm = clip_grad_norm(ps, 1.0)
    assert not torch.isfinite(norm)
    assert torch.equal(ps[1].grad, before)


def test_no_grads():
    assert clip_grad_norm([torch.nn.Parameter(torch.zeros(2))], 1.0).item() == 0.0


def test_clipped_norm_equals_max_norm():
    ps = _params()
    clip_grad_norm(ps, 0.5)
    total = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in ps]))
    assert total.item() == pytest.approx(0.5, abs=1e-4)
