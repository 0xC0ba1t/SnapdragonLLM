import pytest
import torch

from snapdragon.optim.muon import newton_schulz


def _sv(x):
    return torch.linalg.svdvals(x.float())


@pytest.mark.parametrize("shape", [(32, 32), (16, 64), (64, 16)])
def test_singular_values_are_pushed_towards_one(shape):
    torch.manual_seed(0)
    g = torch.randn(*shape)
    out = newton_schulz(g)
    assert out.shape == g.shape
    s_in, s_out = _sv(g), _sv(out)
    assert s_in.max() / s_in.min() > 1.5  # the input is genuinely ill-conditioned
    assert s_out.min() > 0.4
    assert s_out.max() < 1.6


def test_output_is_close_to_the_polar_factor():
    torch.manual_seed(1)
    g = torch.randn(24, 40)
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    polar = u @ vh
    out = newton_schulz(g, steps=5).float()
    cos = (out * polar).sum() / (out.norm() * polar.norm())
    assert cos > 0.9


def test_scale_invariance():
    torch.manual_seed(2)
    g = torch.randn(20, 30)
    assert torch.allclose(newton_schulz(g).float(), newton_schulz(100 * g).float(), atol=3e-2)


def test_zero_matrix_stays_finite():
    out = newton_schulz(torch.zeros(8, 8))
    assert torch.isfinite(out.float()).all()


def test_requires_a_matrix_or_stack():
    with pytest.raises(ValueError, match="expects a matrix"):
        newton_schulz(torch.randn(4))
    with pytest.raises(ValueError, match="expects a matrix"):
        newton_schulz(torch.randn(2, 2, 2, 2))


@pytest.mark.parametrize("shape", [(5, 16, 32), (3, 40, 12)])
def test_stack_equals_one_by_one(shape):
    torch.manual_seed(3)
    stack = torch.randn(*shape)
    batched = newton_schulz(stack).float()
    for i in range(shape[0]):
        assert torch.allclose(batched[i], newton_schulz(stack[i]).float(), atol=2e-2)
