import pytest

from snapdragon.config.data import DataConfig
from snapdragon.config.loader import ConfigError, from_dict
from snapdragon.config.model import ModelConfig
from snapdragon.config.optim import OptimConfig
from snapdragon.config.schema import Config


def test_nested_dataclasses_and_lists():
    cfg = from_dict(
        DataConfig,
        {"sources": [{"name": "web", "path": "p", "weight": 2}], "seq_len": 512},
    )
    assert cfg.sources[0].name == "web"
    assert cfg.sources[0].weight == 2.0
    assert isinstance(cfg.sources[0].weight, float)


def test_unknown_key_suggests_fix():
    with pytest.raises(ConfigError, match="did you mean 'n_layers'"):
        from_dict(ModelConfig, {"n_layer": 4})


def test_error_path_is_dotted():
    with pytest.raises(ConfigError, match=r"optim\.scheduler\.warmup_steps"):
        from_dict(Config, {"optim": {"scheduler": {"warmup_steps": "many"}}})


def test_bool_is_not_an_int():
    with pytest.raises(ConfigError, match="expected an int"):
        from_dict(ModelConfig, {"n_layers": True})


def test_tuple_fields():
    cfg = from_dict(OptimConfig, {"adam_betas": [0.8, 0.9]})
    assert cfg.adam_betas == (0.8, 0.9)
    with pytest.raises(ConfigError, match="expected 2 items"):
        from_dict(OptimConfig, {"adam_betas": [0.8]})


def test_partial_dict_keeps_defaults():
    cfg = from_dict(Config, {"model": {"n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "d_model": 64,
                                       "mlp_hidden": 128, "max_seq_len": 2048}})
    assert cfg.model.n_layers == 2
    assert cfg.train.seq_len == 2048
