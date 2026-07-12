import dataclasses

import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.model.lm import CausalLM

BASE = ModelConfig(vocab_size=64, d_model=32, n_layers=4, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def _grads(mode, every=1):
    cfg = dataclasses.replace(BASE, act_ckpt=mode, act_ckpt_every=every)
    torch.manual_seed(0)
    m = CausalLM(cfg, z_loss_coeff=1e-4).train()
    x = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(1))
    y = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(2))
    loss, _, _ = m(x, y)
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in m.named_parameters()}


@pytest.mark.parametrize(("mode", "every"), [("full", 1), ("selective", 2), ("selective", 3)])
def test_checkpointing_does_not_change_loss_or_grads(mode, every):
    ref_loss, ref = _grads("none")
    loss, got = _grads(mode, every)
    assert loss == pytest.approx(ref_loss, abs=1e-6)
    for name in ref:
        assert torch.allclose(got[name], ref[name], atol=1e-6), name


def test_which_blocks_are_checkpointed():
    from snapdragon.model.transformer import Transformer

    def chosen(**kw):
        t = Transformer(dataclasses.replace(BASE, **kw))
        return [i for i in range(4) if t._checkpoint_block(i)]

    assert chosen(act_ckpt="none") == []
    assert chosen(act_ckpt="full") == [0, 1, 2, 3]
    assert chosen(act_ckpt="selective", act_ckpt_every=2) == [0, 2]


def test_eval_mode_never_checkpoints():
    cfg = dataclasses.replace(BASE, act_ckpt="full")
    m = CausalLM(cfg).eval()
    with torch.no_grad():
        assert m(torch.randint(0, 64, (1, 8))).shape == (1, 8, 64)
