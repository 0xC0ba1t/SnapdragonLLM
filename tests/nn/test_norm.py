import pytest
import torch

from snapdragon.nn.norm import RMSNorm


def test_unit_rms_after_norm():
    x = torch.randn(4, 8, 64) * 7 + 3
    y = RMSNorm(64)(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_weight_scales_output():
    norm = RMSNorm(16)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(3, 16)
    assert torch.allclose(norm(x), 2 * RMSNorm(16)(x), atol=1e-6)


def test_matches_torch_rmsnorm():
    x = torch.randn(5, 32)
    ref = torch.nn.RMSNorm(32, eps=1e-6)
    assert torch.allclose(RMSNorm(32, 1e-6)(x), ref(x), atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_keeps_input_dtype(dtype):
    assert RMSNorm(16)(torch.randn(2, 16).to(dtype)).dtype == dtype


def test_scale_invariance():
    x = torch.randn(2, 16)
    n = RMSNorm(16)
    assert torch.allclose(n(x), n(10 * x), atol=1e-4)
