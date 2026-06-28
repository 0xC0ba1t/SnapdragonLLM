import torch
import torch.nn.functional as F

from snapdragon.nn.mlp import SwiGLU


def test_shape_and_no_bias():
    m = SwiGLU(16, 48)
    assert m(torch.randn(2, 5, 16)).shape == (2, 5, 16)
    assert m.gate_up_proj.bias is None
    assert m.down_proj.bias is None


def test_matches_unfused_reference():
    m = SwiGLU(8, 24)
    x = torch.randn(3, 8)
    w = m.gate_up_proj.weight
    gate = x @ w[:24].T
    up = x @ w[24:].T
    ref = (F.silu(gate) * up) @ m.down_proj.weight.T
    assert torch.allclose(m(x), ref, atol=1e-5)


def test_param_count():
    m = SwiGLU(16, 48)
    assert sum(p.numel() for p in m.parameters()) == 3 * 16 * 48


def test_down_proj_is_marked_for_residual_scaling():
    assert SwiGLU(8, 16).down_proj.is_residual_output
