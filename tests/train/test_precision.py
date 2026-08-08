import torch

from snapdragon.train.precision import autocast_context


def test_fp32_is_a_noop():
    a = torch.randn(4, 4)
    with autocast_context("fp32", torch.device("cpu")):
        assert (a @ a).dtype == torch.float32


def test_bf16_autocast_runs_matmul_in_bf16_but_keeps_params_fp32():
    lin = torch.nn.Linear(4, 4)
    with autocast_context("bf16", torch.device("cpu")):
        out = lin(torch.randn(2, 4))
    assert out.dtype == torch.bfloat16
    assert lin.weight.dtype == torch.float32
