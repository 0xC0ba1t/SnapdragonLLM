import pytest
import torch
import torch.nn.functional as F

from snapdragon.config.model import ModelConfig
from snapdragon.model.lm import CausalLM


@pytest.fixture
def lm():
    cfg = ModelConfig(vocab_size=96, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=32)
    torch.manual_seed(0)
    return CausalLM(cfg, z_loss_coeff=0.0, ce_chunk_tokens=7)


def test_loss_matches_plain_cross_entropy_on_full_logits(lm):
    x = torch.randint(0, 96, (2, 12))
    y = torch.randint(0, 96, (2, 12))
    loss, ce, z = lm(x, y)
    logits = lm(x)
    ref = F.cross_entropy(logits.reshape(-1, 96), y.reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-5)
    assert torch.allclose(ce, ref, atol=1e-5)
    assert z.item() > 0


def test_untrained_loss_is_near_log_vocab(lm):
    x = torch.randint(0, 96, (4, 16))
    loss, _, _ = lm(x, torch.randint(0, 96, (4, 16)))
    assert abs(loss.item() - torch.log(torch.tensor(96.0)).item()) < 0.5


def test_backward_reaches_every_parameter(lm):
    x = torch.randint(0, 96, (2, 8))
    loss, _, _ = lm(x, torch.randint(0, 96, (2, 8)))
    loss.backward()
    missing = [n for n, p in lm.named_parameters() if p.grad is None]
    assert missing == []


def test_ignore_index_targets_do_not_contribute(lm):
    x = torch.randint(0, 96, (1, 10))
    y = torch.randint(0, 96, (1, 10))
    y_masked = y.clone()
    y_masked[:, 5:] = -100
    full, _, _ = lm(x, y)
    half, _, _ = lm(x, y_masked)
    logits = lm(x)
    ref = F.cross_entropy(logits[:, :5].reshape(-1, 96), y[:, :5].reshape(-1))
    assert torch.allclose(half, ref, atol=1e-5)
    assert not torch.allclose(full, half)
