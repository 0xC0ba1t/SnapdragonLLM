import dataclasses

from snapdragon.config.model import ModelConfig
from snapdragon.model.factory import build_model, count_parameters
from snapdragon.optim.param_groups import assert_partition, split_params

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def test_partition_covers_all_params_once():
    m = build_model(CFG)
    s = split_params(m)
    assert_partition(m, s)
    assert sum(s.counts().values()) == count_parameters(m)


def test_group_assignment_by_role():
    s = split_params(build_model(CFG))
    assert s.names["transformer.embed.weight"] == "adam_decay"
    assert s.names["transformer.blocks.0.attn.qkv_proj.weight"] == "muon"
    assert s.names["transformer.blocks.1.mlp.down_proj.weight"] == "muon"
    assert s.names["transformer.blocks.0.attn_norm.weight"] == "adam_no_decay"
    assert s.names["transformer.final_norm.weight"] == "adam_no_decay"
    assert s.names["transformer.blocks.0.attn.q_norm.weight"] == "adam_no_decay"


def test_untied_output_matrix_uses_adam():
    m = build_model(dataclasses.replace(CFG, tie_embeddings=False))
    assert split_params(m).names["transformer.lm_head.weight"] == "adam_decay"


def test_frozen_params_are_skipped():
    m = build_model(CFG)
    m.transformer.embed.weight.requires_grad_(False)
    s = split_params(m)
    assert "transformer.embed.weight" not in s.names
    assert_partition(m, s)


def test_most_parameters_go_to_muon():
    c = split_params(build_model(CFG)).counts()
    assert c["muon"] > c["adam_decay"] + c["adam_no_decay"]
