import pytest
import torch

from snapdragon.inference.sampling import (
    NEG_INF,
    SamplingConfig,
    apply_top_k,
    apply_top_p,
    sample_next,
)


def test_config_validation():
    for bad in ({"temperature": -1}, {"top_k": -1}, {"top_p": 0.0}, {"top_p": 1.5}, {"min_p": 1.0},
                {"repetition_penalty": 0}):
        with pytest.raises(ValueError, match="must"):
            SamplingConfig(**bad)
    assert SamplingConfig(temperature=0).greedy


def test_top_k_keeps_exactly_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = apply_top_k(logits, 2)
    assert torch.isfinite(out).sum() == 2
    assert torch.isfinite(out[0, 1])
    assert torch.isfinite(out[0, 3])


def test_top_k_noop_cases():
    logits = torch.randn(2, 6)
    assert torch.equal(apply_top_k(logits, 0), logits)
    assert torch.equal(apply_top_k(logits, 6), logits)
    assert torch.equal(apply_top_k(logits, 100), logits)


def test_top_p_keeps_the_nucleus():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    kept = torch.isfinite(apply_top_p(logits, 0.7))[0]
    assert kept.tolist() == [True, True, False, False]  # 0.5 + 0.3 reaches 0.7
    assert torch.isfinite(apply_top_p(logits, 0.4))[0].tolist() == [True, False, False, False]  # at least one token


def test_top_p_one_is_a_noop_and_rows_are_independent():
    logits = torch.randn(3, 8)
    assert torch.equal(apply_top_p(logits, 1.0), logits)
    skewed = torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    kept = torch.isfinite(apply_top_p(skewed, 0.9)).sum(dim=-1)
    assert kept.tolist() == [1, 3]


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 2.0, 0.3], [5.0, 0.0, 0.0]])
    assert sample_next(logits, SamplingConfig(temperature=0)).tolist() == [1, 0]


def test_sampling_is_reproducible_with_a_generator_and_respects_top_k():
    logits = torch.zeros(1, 50)
    logits[0, 7] = 2.0
    logits[0, 9] = 2.0
    cfg = SamplingConfig(temperature=1.0, top_k=2)
    draws = {int(sample_next(logits, cfg, torch.Generator().manual_seed(s))) for s in range(40)}
    assert draws <= {7, 9}
    a = sample_next(logits, cfg, torch.Generator().manual_seed(3))
    b = sample_next(logits, cfg, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_low_temperature_concentrates_on_the_mode():
    logits = torch.tensor([[1.0, 1.5, 1.2]])
    cfg = SamplingConfig(temperature=0.05)
    g = torch.Generator().manual_seed(0)
    assert all(int(sample_next(logits, cfg, g)) == 1 for _ in range(30))


def test_empirical_distribution_matches_softmax():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    expected = torch.softmax(logits, -1)[0]
    g = torch.Generator().manual_seed(0)
    draws = torch.tensor([int(sample_next(logits, SamplingConfig(temperature=1.0), g)) for _ in range(4000)])
    freq = torch.bincount(draws, minlength=3) / 4000
    assert torch.allclose(freq, expected, atol=0.03)
    assert float("-inf") == NEG_INF


def test_min_p_cutoff_scales_with_the_top_probability():
    from snapdragon.inference.sampling import apply_min_p

    probs = torch.tensor([[0.6, 0.3, 0.08, 0.02]])
    kept = torch.isfinite(apply_min_p(probs.log(), 0.2))[0]  # floor = 0.2 * 0.6 = 0.12
    assert kept.tolist() == [True, True, False, False]
    flat = torch.full((1, 4), 0.25).log()
    assert torch.isfinite(apply_min_p(flat, 0.5)).all()  # flat distribution: nothing is dropped
    assert torch.equal(apply_min_p(flat, 0.0), flat)


def test_repetition_penalty_pushes_seen_tokens_down_whatever_their_sign():
    from snapdragon.inference.sampling import apply_repetition_penalty

    logits = torch.tensor([[2.0, -2.0, 1.0, 0.5]])
    out = apply_repetition_penalty(logits, torch.tensor([[0, 1]]), 2.0)
    assert out.tolist() == [[1.0, -4.0, 1.0, 0.5]]  # seen: 2/2 and -2*2; unseen unchanged
    assert torch.equal(apply_repetition_penalty(logits, torch.tensor([[0]]), 1.0), logits)
    assert torch.equal(apply_repetition_penalty(logits, torch.empty(1, 0, dtype=torch.long), 2.0), logits)


def test_penalty_can_flip_the_greedy_choice():
    logits = torch.tensor([[3.0, 2.9]])
    cfg = SamplingConfig(temperature=0, repetition_penalty=1.5)
    assert int(sample_next(logits, cfg)) == 0
    assert int(sample_next(logits, cfg, history=torch.tensor([[0]]))) == 1


def test_min_p_in_the_full_pipeline_never_samples_below_the_floor():
    logits = torch.tensor([[5.0, 4.9, -3.0, -4.0]])
    cfg = SamplingConfig(temperature=1.0, min_p=0.3)
    g = torch.Generator().manual_seed(0)
    assert {int(sample_next(logits, cfg, g)) for _ in range(200)} <= {0, 1}
