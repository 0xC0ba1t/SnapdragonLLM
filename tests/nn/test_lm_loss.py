import torch

from snapdragon.nn.functional.ce import reference_ce
from snapdragon.nn.lm_loss import lm_loss


def _data():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 12, 16, generator=g, dtype=torch.float64, requires_grad=True)
    w = torch.randn(30, 16, generator=g, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, 30, (2, 12), generator=g)
    return h, w, t


def test_matches_reference_and_splits_components():
    h, w, t = _data()
    out = lm_loss(h, w, t, z_loss_coeff=1e-3, chunk_tokens=5)
    ref_total = reference_ce(h.reshape(-1, 16), w, t.reshape(-1), 1e-3)
    ref_ce = reference_ce(h.reshape(-1, 16), w, t.reshape(-1), 0.0)
    assert torch.allclose(out.loss, ref_total, atol=1e-10)
    assert torch.allclose(out.ce, ref_ce, atol=1e-10)
    assert torch.allclose(out.ce + 1e-3 * out.z, out.loss.detach(), atol=1e-10)


def test_only_total_loss_carries_grad():
    h, w, t = _data()
    out = lm_loss(h, w, t)
    assert out.loss.requires_grad
    assert not out.ce.requires_grad
    out.loss.backward()
    assert h.grad is not None
    assert w.grad is not None


def test_zero_z_coeff_leaves_loss_equal_to_ce_but_still_reports_z():
    h, w, t = _data()
    out = lm_loss(h, w, t, z_loss_coeff=0.0)
    assert torch.allclose(out.loss.detach(), out.ce)
    assert out.z.item() > 0.0  # logsumexp^2 is a useful diagnostic even when it is not penalised
