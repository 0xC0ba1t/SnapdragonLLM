from pathlib import Path

import pytest
import torch

from snapdragon.config.io import load_config
from snapdragon.config.model import ModelConfig
from snapdragon.model.factory import build_model, count_parameters

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SHRINK = ["train.seq_len=64", "data.seq_len=64"]


def test_real_parameter_count_equals_analytic_for_every_preset():
    for name in ("debug-tiny", "snapdragon-125m"):
        cfg = load_config(CONFIGS / "model" / f"{name}.yaml", SHRINK)
        model = build_model(cfg, device="meta")
        assert count_parameters(model) == cfg.model.num_params()


def test_500m_preset_is_in_budget_without_allocating_memory():
    cfg = load_config(CONFIGS / "model" / "snapdragon-500m.yaml", SHRINK)
    model = build_model(cfg, device="meta")
    n = count_parameters(model)
    assert 480e6 < n < 560e6
    assert n == cfg.model.num_params()
    assert next(model.parameters()).device.type == "meta"


def test_build_from_model_config_only():
    m = build_model(ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, mlp_hidden=64,
                                max_seq_len=16))
    assert m.z_loss_coeff == pytest.approx(1e-4)
    assert next(m.parameters()).device.type == "cpu"


def test_trainable_only_count():
    m = build_model(ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, mlp_hidden=64,
                                max_seq_len=16))
    for p in m.transformer.embed.parameters():
        p.requires_grad_(False)
    assert count_parameters(m, trainable_only=True) == count_parameters(m) - 64 * 32
    assert isinstance(next(m.parameters()), torch.nn.Parameter)
