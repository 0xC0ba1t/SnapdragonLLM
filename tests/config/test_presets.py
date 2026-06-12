from pathlib import Path

import pytest

from snapdragon.config.io import load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
MODEL_PRESETS = sorted((CONFIGS / "model").glob("*.yaml"))


@pytest.mark.parametrize("path", MODEL_PRESETS, ids=lambda p: p.stem)
def test_model_preset_loads(path):
    # model presets only set the model section, so train/data seq_len must be shrunk for tiny ones
    cfg = load_config(path, ["train.seq_len=64", "data.seq_len=64"])
    assert cfg.model.num_params() > 0


@pytest.mark.parametrize(
    ("name", "lo", "hi"),
    [("debug-tiny", 0.05e6, 1e6), ("snapdragon-125m", 110e6, 140e6), ("snapdragon-500m", 480e6, 560e6)],
)
def test_param_budget(name, lo, hi):
    cfg = load_config(CONFIGS / "model" / f"{name}.yaml", ["train.seq_len=64", "data.seq_len=64"])
    assert lo < cfg.model.num_params() < hi
