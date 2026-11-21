import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.config.optim import OptimConfig
from snapdragon.distributed.fsdp import build_fsdp_optimizer
from snapdragon.distributed.testing import gloo_available, run_distributed
from snapdragon.distributed.workers import fsdp_matches_single_process
from snapdragon.model.factory import build_model

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def test_fsdp_optimizer_covers_every_parameter_once_with_decay_split():
    m = build_model(CFG)
    opt = build_fsdp_optimizer(m, OptimConfig(adam_weight_decay=0.1))
    owned = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(owned) == len(set(owned)) == len(list(m.parameters()))
    assert sorted(g["weight_decay"] for g in opt.param_groups) == [0.0, 0.1]
    norm_ids = {id(p) for n, p in m.named_parameters() if "norm" in n}
    no_decay = next(g for g in opt.param_groups if g["weight_decay"] == 0.0)
    assert {id(p) for p in no_decay["params"]} == norm_ids


@pytest.mark.slow
@pytest.mark.skipif(not gloo_available(), reason="no usable gloo transport on this platform")
def test_sharded_training_step_matches_unsharded():
    losses = run_distributed(fsdp_matches_single_process, 2)
    ref_loss, ref_weight_sum = losses[0]["reference"]
    for r in losses:
        assert r["loss"] == pytest.approx(ref_loss, rel=1e-4)
        assert r["weight_sum"] == pytest.approx(ref_weight_sum, rel=1e-3)
    assert isinstance(torch.tensor(0.0), torch.Tensor)


def test_mesh_device_type_follows_the_parameters_not_cuda_availability(monkeypatch):
    from snapdragon.distributed.fsdp import mesh_device_type

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert mesh_device_type(torch.nn.Linear(2, 2)) == "cpu"
    assert mesh_device_type(torch.nn.Module()) == "cpu"
