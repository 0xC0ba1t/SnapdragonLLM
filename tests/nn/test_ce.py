import pytest
import torch

from snapdragon.nn.functional.ce import IGNORE_INDEX, chunked_ce_checkpointed, reference_ce


def _data(n=50, d=16, v=40, dtype=torch.float64, ignore=0):
    g = torch.Generator().manual_seed(0)
    h = torch.randn(n, d, generator=g, dtype=dtype, requires_grad=True)
    w = torch.randn(v, d, generator=g, dtype=dtype, requires_grad=True)
    t = torch.randint(0, v, (n,), generator=g)
    t[:ignore] = IGNORE_INDEX
    return h, w, t


def test_reference_matches_f_cross_entropy_without_z_loss():
    h, w, t = _data()
    ref = torch.nn.functional.cross_entropy((h @ w.T), t)
    assert torch.allclose(reference_ce(h, w, t), ref)


def test_z_loss_adds_positive_term():
    h, w, t = _data()
    assert reference_ce(h, w, t, 1e-2) > reference_ce(h, w, t, 0.0)


@pytest.mark.parametrize("z", [0.0, 1e-4])
@pytest.mark.parametrize("ignore", [0, 7])
def test_chunked_matches_reference_forward_and_backward(z, ignore):
    h, w, t = _data(ignore=ignore)
    ref = reference_ce(h, w, t, z)
    ref.backward()
    gh, gw = h.grad.clone(), w.grad.clone()
    h.grad = w.grad = None
    out = chunked_ce_checkpointed(h, w, t, z, chunk_tokens=16)
    out.backward()
    assert torch.allclose(out, ref, atol=1e-10)
    assert torch.allclose(h.grad, gh, atol=1e-10)
    assert torch.allclose(w.grad, gw, atol=1e-10)


def test_all_ignored_is_zero_not_nan():
    h, w, t = _data(n=8, ignore=8)
    out = chunked_ce_checkpointed(h, w, t, 1e-4, chunk_tokens=4)
    assert out.item() == 0.0
