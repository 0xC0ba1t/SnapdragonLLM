import pytest
import torch

from snapdragon.optim.quant import (
    dequantize_blockwise,
    linear_codebook,
    quantize_blockwise,
)


def test_codebooks():
    s, u = linear_codebook(True), linear_codebook(False)
    assert s.numel() == u.numel() == 256
    assert s.min() == -1
    assert s.max() == 1
    assert (s == 0).any()
    assert u.min() == 0
    assert u.max() == 1
    assert torch.all(s[1:] >= s[:-1])


@pytest.mark.parametrize("shape", [(4096,), (3, 1000), (7,), (2048,), (2049,)])
def test_roundtrip_error_is_bounded_by_half_a_step(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    cb = linear_codebook(True)
    codes, absmax = quantize_blockwise(x, cb, block=512)
    y = dequantize_blockwise(codes, absmax, cb, x.shape, block=512)
    assert y.shape == x.shape
    assert codes.dtype == torch.uint8
    step = 2 / 254  # codebook spacing in normalised units
    bound = absmax.max() * step / 2 * 1.01
    assert (x - y).abs().max() <= bound


def test_blocks_are_independent_outlier_only_hurts_its_block():
    x = torch.randn(4096)
    x[5] = 1000.0  # outlier in block 0 of size 1024
    cb = linear_codebook(True)
    codes, absmax = quantize_blockwise(x, cb, block=1024)
    y = dequantize_blockwise(codes, absmax, cb, x.shape, block=1024)
    clean = (x[1024:] - y[1024:]).abs().max()
    assert clean < 0.02  # untouched blocks keep fine resolution
    assert (x[:1024] - y[:1024]).abs().max() > 1.0  # the outlier block lost it


def test_zeros_roundtrip_exactly_and_do_not_nan():
    x = torch.zeros(5000)
    cb = linear_codebook(True)
    codes, absmax = quantize_blockwise(x, cb)
    y = dequantize_blockwise(codes, absmax, cb, x.shape)
    assert torch.equal(y, x)


def test_unsigned_codebook_for_nonnegative_state():
    x = torch.rand(3000)
    cb = linear_codebook(False)
    codes, absmax = quantize_blockwise(x, cb, block=1000)
    y = dequantize_blockwise(codes, absmax, cb, x.shape, block=1000)
    assert (x - y).abs().max() < 0.003
    assert (y >= 0).all()


def test_storage_is_about_one_byte_per_value():
    x = torch.randn(1 << 16)
    codes, absmax = quantize_blockwise(x, linear_codebook(True))
    nbytes = codes.numel() * codes.element_size() + absmax.numel() * absmax.element_size()
    assert nbytes < 1.01 * x.numel()
