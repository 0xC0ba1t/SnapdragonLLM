import pytest

from snapdragon.config.model import ModelConfig
from snapdragon.model.flops import flops_per_token, mfu, peak_tflops_for

CFG = ModelConfig()  # the 500M


def test_close_to_six_n_rule_of_thumb():
    f = flops_per_token(CFG, 2048)
    six_n = 6 * CFG.num_params()
    assert 0.9 * six_n < f < 1.3 * six_n  # attention adds a modest extra at 2k context


def test_longer_context_costs_more_per_token():
    assert flops_per_token(CFG, 4096) > flops_per_token(CFG, 1024)


def test_recompute_adds_cost_in_order():
    a, b, c = (flops_per_token(CFG, 2048, r) for r in ("none", "selective", "full"))
    assert a < b < c
    assert c < 1.4 * a  # full recompute is roughly +33%, not +100%


def test_mfu_is_a_fraction():
    # 4090-class peak 165 TFLOPs; 12k tok/s on a 500M model is ~ 38 TFLOPs -> ~23%
    m = mfu(12_000, CFG, 165.0, 2048)
    assert 0.1 < m < 0.4


def test_peak_lookup_prefers_longest_match():
    assert peak_tflops_for("NVIDIA GeForce RTX 4070 Ti SUPER") == 80.0
    assert peak_tflops_for("NVIDIA GeForce RTX 4070") == 58.0
    assert peak_tflops_for("mystery gpu") is None


def test_unknown_recompute_mode():
    with pytest.raises(KeyError):
        flops_per_token(CFG, 2048, "everything")
