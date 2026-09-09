import pytest
import torch

from snapdragon.distributed.env import DistEnv
from snapdragon.distributed.testing import gloo_available, run_distributed
from snapdragon.distributed.workers import ddp_grads, single_process_grads
from snapdragon.distributed.wrap import wrap_ddp, wrap_model


def test_single_process_wrap_is_identity():
    m = torch.nn.Linear(2, 2)
    assert wrap_ddp(m, DistEnv()) is m
    assert wrap_model(m, DistEnv(), "ddp") is m


def test_unknown_parallel_mode():
    with pytest.raises(ValueError, match="parallel must be"):
        wrap_model(torch.nn.Linear(2, 2), DistEnv(), "zero3")


@pytest.mark.slow
@pytest.mark.skipif(not gloo_available(), reason="no usable gloo transport on this platform")
def test_ddp_gradients_equal_single_process_full_batch_gradients():
    ranks = run_distributed(ddp_grads, 2)
    ref = single_process_grads()
    for name in ref:
        # both ranks hold the same averaged gradient...
        assert torch.allclose(ranks[0][name], ranks[1][name], atol=1e-6), name
        # ...which equals the gradient of the whole batch on one process
        assert torch.allclose(ranks[0][name], ref[name], atol=1e-5), name
