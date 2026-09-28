import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.inference.kv_cache import KVCache
from snapdragon.model.factory import build_model

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=32)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_model(CFG).eval()
    return m


def _tokens(batch=2, length=14, seed=1):
    return torch.randint(0, 64, (batch, length), generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def test_prefill_with_cache_equals_the_plain_forward(model):
    x = _tokens()
    cache = KVCache(CFG, batch_size=2)
    cached = model(x, cache=cache)
    assert cache.pos == 14
    assert torch.allclose(cached, model(x), atol=1e-5)


@torch.no_grad()
def test_token_by_token_decoding_matches_the_full_sequence(model):
    x = _tokens()
    full = model(x)
    cache = KVCache(CFG, 2)
    for t in range(x.shape[1]):
        step = model(x[:, t : t + 1], cache=cache)
        assert torch.allclose(step[:, 0], full[:, t], atol=1e-5), f"position {t}"
    assert cache.pos == x.shape[1]


@torch.no_grad()
def test_chunked_prefill_then_decode(model):
    x = _tokens(length=16)
    full = model(x)
    cache = KVCache(CFG, 2)
    out = [model(x[:, :6], cache=cache), model(x[:, 6:13], cache=cache)]  # second chunk exercises the mask
    out.append(model(x[:, 13:14], cache=cache))
    out.append(model(x[:, 14:16], cache=cache))
    assert torch.allclose(torch.cat(out, dim=1), full, atol=1e-5)


@torch.no_grad()
def test_positions_continue_from_the_cache(model):
    """Without correct RoPE offsets the second chunk would be rotated as if it started at 0."""
    x = _tokens(batch=1, length=10)
    cache = KVCache(CFG, 1)
    model(x[:, :5], cache=cache)
    wrong_cache = KVCache(CFG, 1)
    model(x[:, :5], cache=wrong_cache)
    right = model(x[:, 5:], cache=cache)
    shifted = model(x[:, 5:], position_ids=torch.arange(5).expand(1, 5), cache=wrong_cache)
    assert not torch.allclose(right, shifted, atol=1e-4)


@torch.no_grad()
def test_cache_in_bf16_stays_close(model):
    x = _tokens()
    cache = KVCache(CFG, 2, dtype=torch.bfloat16)
    assert torch.allclose(model(x, cache=cache), model(x), atol=0.1)


def test_causal_lm_forwards_the_cache(model):
    from snapdragon.model.lm import CausalLM

    lm = CausalLM(CFG).eval()
    cache = KVCache(CFG, 1)
    with torch.no_grad():
        out = lm(_tokens(1, 6), cache=cache)
    assert out.shape == (1, 6, 64)
    assert cache.pos == 6
