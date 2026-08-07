import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.config.optim import OptimConfig
from snapdragon.model.factory import build_model
from snapdragon.optim.factory import build_optimizer
from snapdragon.optim.memory import estimate_state_bytes, state_bytes

CFG = ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=192, max_seq_len=32)


def _step(model, opt):
    x = torch.randint(0, 512, (2, 16))
    loss, _, _ = model(x, x)
    loss.backward()
    opt.step()


@pytest.mark.parametrize("eight_bit", [True, False])
def test_estimate_matches_measured_bytes(eight_bit):
    cfg = OptimConfig(eight_bit=eight_bit)
    model = build_model(CFG)
    opt = build_optimizer(model, cfg)
    _step(model, opt)
    est = estimate_state_bytes(model, cfg)
    adam_measured = state_bytes(opt.optimizers["adam"])
    muon_measured = state_bytes(opt.optimizers["muon"])
    assert est["muon"] == muon_measured
    assert abs(est["adam"] - adam_measured) <= 0.001 * est["adam"] + 8  # step counters are python ints


def test_eight_bit_saves_most_of_adams_memory():
    model = build_model(CFG, device="meta")
    small = estimate_state_bytes(model, OptimConfig(eight_bit=True))["adam"]
    full = estimate_state_bytes(model, OptimConfig(eight_bit=False))["adam"]
    assert small < 0.35 * full


def test_500m_state_fits_in_about_a_gigabyte_and_a_half():
    from snapdragon.config.model import ModelConfig as MC

    model = build_model(MC(), device="meta")
    est = estimate_state_bytes(model, OptimConfig())
    gb = est["total"] / 1e9
    assert 1.0 < gb < 1.6  # vs ~4.2 GB for fp32 AdamW on the same model
