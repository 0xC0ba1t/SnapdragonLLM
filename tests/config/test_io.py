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
