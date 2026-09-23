import math

import pytest

from snapdragon.config.model import ModelConfig
from snapdragon.eval.perplexity import PerplexityResult, bytes_per_token, evaluate_perplexity
from snapdragon.model.factory import build_model


def test_result_conversions():
    r = PerplexityResult(loss=math.log(8.0), tokens=100)
    assert r.ppl == pytest.approx(8.0)
    assert r.bits_per_token == pytest.approx(3.0)
    assert r.bits_per_byte(bytes_per_token=3.0) == pytest.approx(1.0)


def test_huge_loss_does_not_overflow():
    assert PerplexityResult(loss=1e6, tokens=1).ppl < float("inf")


def test_untrained_model_scores_near_uniform(tmp_path, make_token_dataset):
    # tokens 0..99 are all < vocab (128) except the eos id 65000, which we avoid via a small corpus
    ds = make_token_dataset(tmp_path, 600)
    cfg = ModelConfig(vocab_size=65001, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, mlp_hidden=32,
                      max_seq_len=32)
    model = build_model(cfg)
    r = evaluate_perplexity(model, ds, seq_len=16, batch_size=2, max_batches=5)
    assert r.tokens == 5 * 2 * 16
    assert abs(r.loss - math.log(65001)) < 1.0


def test_bytes_per_token(small_tokenizer):
    value = bytes_per_token(small_tokenizer, ["abc def ghi jab cde fgh"] * 10)
    assert value > 1.0  # merges make tokens longer than one byte on text it was trained on
    assert bytes_per_token(small_tokenizer, []) == 0.0
