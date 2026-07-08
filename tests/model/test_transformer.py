import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.model.transformer import Transformer


@pytest.fixture
def cfg():
    return ModelConfig(vocab_size=96, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=32)


def test_forward_shape(cfg):
    m = Transformer(cfg)
    h = m(torch.randint(0, 96, (2, 10)))
    assert h.shape == (2, 10, 32)
    assert m.logits(h).shape == (2, 10, 96)


def test_param_count_matches_analytic_formula(cfg):
    n = sum(p.numel() for p in Transformer(cfg).parameters())
    assert n == cfg.num_params()


def test_untied_param_count():
    cfg = ModelConfig(vocab_size=96, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64,
                      max_seq_len=32, tie_embeddings=False)
    m = Transformer(cfg)
    assert sum(p.numel() for p in m.parameters()) == cfg.num_params()
    assert m.output_weight is m.lm_head.weight


def test_tied_output_weight_is_the_embedding(cfg):
    m = Transformer(cfg)
    assert m.output_weight is m.embed.weight


def test_sequence_longer_than_max_is_rejected(cfg):
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        Transformer(cfg)(torch.zeros(1, 33, dtype=torch.long))


def test_model_is_causal(cfg):
    m = Transformer(cfg).eval()
    a = torch.randint(0, 96, (1, 16))
    b = a.clone()
    b[:, 10:] = torch.randint(0, 96, (1, 6))
    with torch.no_grad():
        ha, hb = m(a), m(b)
    assert torch.allclose(ha[:, :10], hb[:, :10], atol=1e-5)
