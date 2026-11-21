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


class _CountMM(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default):
            self.n += 1
        return func(*args, **(kwargs or {}))


def _mm_calls(mode):
    cfg = dataclasses.replace(BASE, act_ckpt=mode)
    torch.manual_seed(0)
    m = CausalLM(cfg).train()
    x = torch.randint(0, 64, (2, 12))
    with _CountMM() as counter:
        loss, _, _ = m(x, x)
        loss.backward()
    return counter.n


def test_sac_matches_reference_grads():
    ref_loss, ref = _grads("none")
    loss, got = _grads("sac")
    assert loss == pytest.approx(ref_loss, abs=1e-6)
    for name in ref:
        assert torch.allclose(got[name], ref[name], atol=1e-6), name


def test_sac_recomputes_no_matmuls_unlike_full():
    none, sac, full = _mm_calls("none"), _mm_calls("sac"), _mm_calls("full")
    assert sac == none  # matmul outputs were saved, nothing redone
    assert full > none  # full checkpointing redoes every block's forward matmuls


def test_sac_never_saves_bmm_so_unfused_attention_scores_are_recomputed():
    """Regression: with the math SDPA fallback the T x T scores are bmm outputs. Saving them OOMed a
    16 GB T4 on the 500M preset (28 layers x 4 x 20 x 2048^2 values)."""
    from torch.utils.checkpoint import CheckpointPolicy

    from snapdragon.model.transformer import _SAVE, _sac_policy

    assert torch.ops.aten.bmm.default not in _SAVE
    assert _sac_policy(None, torch.ops.aten.bmm.default) == CheckpointPolicy.PREFER_RECOMPUTE
    assert _sac_policy(None, torch.ops.aten.mm.default) == CheckpointPolicy.MUST_SAVE
    assert _sac_policy(None, torch.ops.aten.addmm.default) == CheckpointPolicy.MUST_SAVE


def test_sac_with_forced_math_attention_matches_unchecked_gradients():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    ref_loss, ref = _grads("none")
    with sdpa_kernel(SDPBackend.MATH):
        loss, got = _grads("sac")
    assert loss == pytest.approx(ref_loss, abs=1e-6)
    for name in ref:
        assert torch.allclose(got[name], ref[name], atol=1e-6), name
