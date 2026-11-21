import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.model.compile import compile_blocks, should_compile, unwrap_model
from snapdragon.model.lm import CausalLM

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def test_should_compile_settings():
    assert should_compile("true") is True
    assert should_compile("false") is False
    assert isinstance(should_compile("auto"), bool)
    with pytest.raises(ValueError, match="auto, true or false"):
        should_compile("maybe")


def test_compile_blocks_keeps_keys_and_outputs():
    torch.manual_seed(0)
    m = CausalLM(CFG).eval()
    keys = list(m.state_dict())
    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        ref = m(x)
    assert compile_blocks(m, backend="eager") == 3
    assert list(m.state_dict()) == keys
    with torch.no_grad():
        assert torch.allclose(m(x), ref, atol=1e-5)


def test_unwrap_handles_nested_wrappers():
    class Fake(torch.nn.Module):
        def __init__(self, inner, attr):
            super().__init__()
            setattr(self, attr, inner)

    core = torch.nn.Linear(2, 2)
    assert unwrap_model(Fake(Fake(core, "_orig_mod"), "module")) is core
    assert unwrap_model(core) is core


@pytest.mark.parametrize("act_ckpt", ["none", "selective", "full"])
def test_compiled_blocks_train_identically_to_eager(make_trainer, tiny_config, act_ckpt):
    """Regional compile (eager backend here, so no C++ toolchain is needed) must not change the numbers,
    including the backward pass through checkpointed blocks."""
    cfg = tiny_config()
    cfg.model.act_ckpt = act_ckpt
    plain = make_trainer(cfg)
    compiled = make_trainer(cfg)
    assert compile_blocks(compiled.raw_model, backend="eager") == cfg.model.n_layers
    for _ in range(3):
        a = plain.train_step()["loss"]
        b = compiled.train_step()["loss"]
        assert b == pytest.approx(a, abs=1e-5)
    for (n, p), (_, q) in zip(plain.raw_model.named_parameters(), compiled.raw_model.named_parameters()):
        assert torch.allclose(p, q, atol=1e-5), n


def test_sac_skips_compilation_with_a_warning_and_still_trains(make_trainer, tiny_config):
    """Regression: compiled blocks inside selective checkpointing crashed backward on a Colab T4
    ("aten.pow.Tensor_Scalar invocation index 3 ... not found in storage")."""
    cfg = tiny_config()
    cfg.model.act_ckpt = "sac"
    t = make_trainer(cfg)
    with pytest.warns(UserWarning, match="sac"):
        assert compile_blocks(t.raw_model, backend="eager") == 0
    assert not any(hasattr(b, "_compiled_call_impl") and b._compiled_call_impl is not None
                   for b in t.raw_model.transformer.blocks)
    assert t.train_step()["loss"] > 0
