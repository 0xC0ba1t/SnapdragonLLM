import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.inference.kv_cache import KVCache

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def test_shape_and_memory():
    c = KVCache(CFG, batch_size=2)
    assert c.k.shape == (3, 2, 2, 16, 8)
    assert c.nbytes == 2 * 3 * 2 * 2 * 16 * 8 * 4


def test_update_returns_filled_prefix_only():
    c = KVCache(CFG, 1)
    k = torch.arange(4 * 2 * 8, dtype=torch.float32).view(1, 2, 4, 8)
    ks, vs = c.update(0, k, -k)
    assert ks.shape == (1, 2, 4, 8)
    assert torch.equal(ks, k)
    assert torch.equal(vs, -k)
    c.advance(4)
    k2 = torch.ones(1, 2, 1, 8)
    ks2, _ = c.update(0, k2, k2)
    assert ks2.shape[2] == 5
    assert torch.equal(ks2[:, :, :4], k)  # earlier entries are preserved
    assert torch.equal(ks2[:, :, 4:], k2)


def test_layers_are_independent():
    c = KVCache(CFG, 1)
    c.update(0, torch.ones(1, 2, 2, 8), torch.ones(1, 2, 2, 8))
    other, _ = c.update(1, torch.zeros(1, 2, 2, 8), torch.zeros(1, 2, 2, 8))
    assert other.abs().sum() == 0


def test_overflow_is_an_error():
    c = KVCache(CFG, 1, max_len=4)
    c.advance(3)
    with pytest.raises(ValueError, match="overflow"):
        c.update(0, torch.zeros(1, 2, 2, 8), torch.zeros(1, 2, 2, 8))


def test_cannot_be_longer_than_the_model_context():
    with pytest.raises(ValueError, match="max_seq_len"):
        KVCache(CFG, 1, max_len=32)


def test_reset():
    c = KVCache(CFG, 1)
    c.advance(5)
    c.reset()
    assert c.pos == 0


def test_dtype_conversion_on_write():
    c = KVCache(CFG, 1, dtype=torch.bfloat16)
    ks, _ = c.update(0, torch.ones(1, 2, 1, 8), torch.ones(1, 2, 1, 8))
    assert ks.dtype == torch.bfloat16
