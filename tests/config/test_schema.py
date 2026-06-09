import pytest

from snapdragon.config.schema import Config


def test_default_config_is_valid():
    cfg = Config()
    assert cfg.model.max_seq_len == cfg.train.seq_len == cfg.data.seq_len


def test_seq_len_longer_than_model_rejected():
    cfg = Config()
    cfg.train.seq_len = 4096
    with pytest.raises(ValueError, match="exceeds"):
        cfg.validate()


def test_seq_len_mismatch_rejected():
    cfg = Config()
    cfg.data.seq_len = 1024
    with pytest.raises(ValueError, match="must equal"):
        cfg.validate()


def test_doc_masking_needs_flex():
    cfg = Config()
    cfg.data.doc_masking = True
    with pytest.raises(ValueError, match="flex"):
        cfg.validate()
