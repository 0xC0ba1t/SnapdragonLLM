import pytest
import torch

from snapdragon.utils.device import autocast_dtype, get_device


def test_explicit_device():
    assert get_device("cpu") == torch.device("cpu")


def test_auto_device_returns_something():
    assert get_device().type in {"cpu", "cuda", "mps"}


@pytest.mark.parametrize(
    ("name", "dtype"), [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)]
)
def test_autocast_dtype(name, dtype):
    assert autocast_dtype(name) is dtype


def test_autocast_dtype_rejects_unknown():
    with pytest.raises(ValueError, match="unknown precision"):
        autocast_dtype("int4")


def test_configure_backends_sets_allocator_and_precision_without_clobbering_user_choice():
    from snapdragon.utils.device import configure_backends

    env = {}
    applied = configure_backends(env)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert applied["matmul_precision"] == "high"

    user = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
    assert configure_backends(user)["alloc_conf"] == "max_split_size_mb:128"
