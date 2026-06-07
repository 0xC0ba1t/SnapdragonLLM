import pytest

from snapdragon.config.data import DataConfig, SourceConfig


def test_weights_normalise():
    cfg = DataConfig(sources=[SourceConfig("a", "x", 3.0), SourceConfig("b", "y", 1.0)])
    w = cfg.normalized_weights()
    assert w == {"a": 0.75, "b": 0.25}


def test_duplicate_source_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        DataConfig(sources=[SourceConfig("a", "x"), SourceConfig("a", "y")])


def test_nonpositive_weight_rejected():
    with pytest.raises(ValueError, match="positive"):
        SourceConfig("a", "x", 0.0)
