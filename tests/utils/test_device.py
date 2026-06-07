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
    with pytest.raises(ValueError):
        autocast_dtype("int4")
