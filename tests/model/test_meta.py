import torch

from snapdragon.config.model import ModelConfig
from snapdragon.model.factory import build_model
from snapdragon.model.meta import materialize

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def test_meta_then_materialize_equals_direct_build():
    torch.manual_seed(7)
    direct = build_model(CFG)
    torch.manual_seed(7)
    lazy = materialize(build_model(CFG, device="meta"), "cpu")
    for (n1, p1), (n2, p2) in zip(direct.named_parameters(), lazy.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1
    x = torch.randint(0, 64, (1, 8))
    assert torch.allclose(direct(x), lazy(x), atol=1e-6)


def test_materialized_model_has_no_meta_tensors():
    m = materialize(build_model(CFG, device="meta"))
    assert all(p.device.type == "cpu" for p in m.parameters())
    assert all(b.device.type == "cpu" for b in m.buffers())
    assert torch.isfinite(m.transformer.rope.cos).all()
