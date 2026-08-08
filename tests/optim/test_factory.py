import dataclasses

import torch

from snapdragon.config.model import ModelConfig
from snapdragon.config.optim import OptimConfig
from snapdragon.model.factory import build_model
from snapdragon.optim.adamw8bit import AdamW8bit
from snapdragon.optim.factory import build_optimizer
from snapdragon.optim.muon import Muon

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def test_builds_muon_and_adam():
    opt = build_optimizer(build_model(CFG), OptimConfig())
    assert isinstance(opt.optimizers["muon"], Muon)
    assert isinstance(opt.optimizers["adam"], AdamW8bit)


def test_adam_groups_have_the_right_weight_decay():
    cfg = OptimConfig(adam_weight_decay=0.1)
    opt = build_optimizer(build_model(CFG), cfg)
    decays = sorted(g["weight_decay"] for g in opt.optimizers["adam"].param_groups)
    assert decays == [0.0, 0.1]


def test_learning_rates_come_from_config():
    opt = build_optimizer(build_model(CFG), OptimConfig(muon_lr=0.03, adam_lr=1e-3))
    assert opt.optimizers["muon"].param_groups[0]["lr"] == 0.03
    assert all(g["lr"] == 1e-3 for g in opt.optimizers["adam"].param_groups)


def test_can_fall_back_to_fp32_adamw():
    opt = build_optimizer(build_model(CFG), OptimConfig(eight_bit=False))
    assert isinstance(opt.optimizers["adam"], torch.optim.AdamW)


def test_one_real_training_step_changes_every_parameter():
    model = build_model(CFG)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = build_optimizer(model, OptimConfig())
    x = torch.randint(0, 64, (2, 8))
    loss, _, _ = model(x, torch.randint(0, 64, (2, 8)))
    loss.backward()
    opt.step()
    unchanged = [n for n, p in model.named_parameters() if torch.equal(p, before[n])]
    assert unchanged == []


def test_untied_model_gets_its_head_in_adam():
    m = build_model(dataclasses.replace(CFG, tie_embeddings=False))
    opt = build_optimizer(m, OptimConfig())
    adam_params = {id(p) for g in opt.optimizers["adam"].param_groups for p in g["params"]}
    assert id(m.transformer.lm_head.weight) in adam_params


def test_offload_flag_routes_state_to_host():
    opt = build_optimizer(build_model(CFG), OptimConfig(offload_states=True))
    assert opt.optimizers["muon"].param_groups[0]["state_device"] == torch.device("cpu")
    assert opt.optimizers["adam"].param_groups[0]["state_device"] == torch.device("cpu")
