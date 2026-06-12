import pytest

from snapdragon.config.overrides import apply_overrides, parse_override


def test_parse_types():
    assert parse_override("train.seed=7") == (["train", "seed"], 7)
    assert parse_override("optim.eight_bit=false") == (["optim", "eight_bit"], False)
    assert parse_override("train.run_name=abc") == (["train", "run_name"], "abc")


def test_scientific_notation_is_a_float():
    out = apply_overrides({}, ["optim.adam_lr=3e-4"])
    assert out["optim"]["adam_lr"] == pytest.approx(3e-4)


def test_apply_creates_sections_and_does_not_mutate():
    base = {"train": {"seed": 1}}
    out = apply_overrides(base, ["train.seed=2", "optim.scheduler.kind=cosine"])
    assert out == {"train": {"seed": 2}, "optim": {"scheduler": {"kind": "cosine"}}}
    assert base == {"train": {"seed": 1}}


def test_malformed_override():
    with pytest.raises(ValueError, match="section.key=value"):
        parse_override("nonsense")
    with pytest.raises(ValueError, match="empty path"):
        parse_override("a..b=1")


def test_cannot_descend_into_scalar():
    with pytest.raises(ValueError, match="not a section"):
        apply_overrides({"train": {"seed": 1}}, ["train.seed.x=1"])
