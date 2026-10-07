import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.inference.generate import generate, stream_generate
from snapdragon.inference.sampling import SamplingConfig
from snapdragon.model.factory import build_model

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=32)
GREEDY = SamplingConfig(temperature=0)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(CFG).eval()


@torch.no_grad()
def _uncached_greedy(model, prompt, n):
    """Reference: re-run the whole sequence for every new token (slow, but obviously correct)."""
    ids = list(prompt)
    for _ in range(n):
        logits = model(torch.tensor([ids]))[0, -1]
        ids.append(int(logits.argmax()))
    return ids[len(prompt):]


def test_cached_greedy_generation_equals_the_naive_uncached_loop(model):
    prompt = [3, 14, 15, 9, 2]
    assert list(stream_generate(model, prompt, 12, GREEDY)) == _uncached_greedy(model, prompt, 12)


def test_same_seed_same_text_different_seed_different_text(model):
    cfg = SamplingConfig(temperature=1.0)
    a = generate(model, [1, 2, 3], 16, cfg, seed=5).token_ids
    b = generate(model, [1, 2, 3], 16, cfg, seed=5).token_ids
    c = generate(model, [1, 2, 3], 16, cfg, seed=6).token_ids
    assert a == b
    assert a != c


def test_stops_on_a_stop_token_and_reports_why(model):
    first = _uncached_greedy(model, [1, 2, 3], 1)[0]
    res = generate(model, [1, 2, 3], 10, GREEDY, stop_ids=[first])
    assert res.token_ids == [first]
    assert res.finish_reason == "stop"
    assert generate(model, [1, 2, 3], 4, GREEDY).finish_reason == "length"


def test_length_limit(model):
    assert len(list(stream_generate(model, [1], 7, GREEDY))) == 7


def test_long_prompts_are_left_truncated_to_fit(model):
    long_prompt = list(range(40)) * 2  # 80 tokens, context is 32
    out = list(stream_generate(model, long_prompt, 8, GREEDY))
    assert len(out) == 8
    assert out == list(stream_generate(model, long_prompt[-24:], 8, GREEDY))


def test_errors(model):
    with pytest.raises(ValueError, match="must not be empty"):
        list(stream_generate(model, [], 4))
    with pytest.raises(ValueError, match="leaves no room"):
        list(stream_generate(model, [1], 32))


def test_streaming_is_lazy_and_restores_the_training_flag(model):
    model.train()
    it = stream_generate(model, [1, 2], 5, GREEDY)
    first = next(it)
    assert isinstance(first, int)
    assert not model.training  # eval while generating
    list(it)
    assert model.training  # restored afterwards
    model.eval()


def test_generation_uses_the_cache_not_a_full_recompute(model, monkeypatch):
    seen = []
    original = model.transformer.forward

    def spy(tokens, *a, **kw):
        seen.append(tokens.shape[1])
        return original(tokens, *a, **kw)

    monkeypatch.setattr(model.transformer, "forward", spy)
    list(stream_generate(model, [1, 2, 3, 4], 5, GREEDY))
    assert seen == [4, 1, 1, 1, 1]  # one prefill of the prompt, then single-token steps
