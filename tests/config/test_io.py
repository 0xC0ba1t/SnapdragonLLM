import pytest

from snapdragon.config.io import config_to_dict, dump_config, load_config, read_yaml
from snapdragon.config.schema import Config


def test_roundtrip_through_yaml(tmp_path):
    cfg = Config()
    cfg.optim.adam_betas = (0.85, 0.95)
    path = tmp_path / "c.yaml"
    dump_config(cfg, path)
    assert load_config(path) == cfg


def test_dict_has_no_tuples():
    d = config_to_dict(Config())
    assert d["optim"]["adam_betas"] == [0.9, 0.95]


def test_empty_file_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(path) == Config()


def test_non_mapping_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        read_yaml(path)


def test_load_with_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    dump_config(Config(), path)
    cfg = load_config(path, ["train.micro_batch_size=2", "optim.adam_lr=1e-3"])
    assert cfg.train.micro_batch_size == 2
    assert cfg.optim.adam_lr == pytest.approx(1e-3)


def test_extends_merges_deeply(tmp_path):
    (tmp_path / "base.yaml").write_text("train:\n  seed: 5\n  micro_batch_size: 8\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ntrain:\n  micro_batch_size: 2\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert (cfg.train.seed, cfg.train.micro_batch_size) == (5, 2)


def test_circular_extends_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")
