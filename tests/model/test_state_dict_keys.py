"""Checkpoint compatibility guard: renaming a module silently breaks every saved checkpoint, so the
key layout is pinned here. Change this list only together with a checkpoint migration."""
from snapdragon.config.model import ModelConfig
from snapdragon.model.factory import build_model

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)

EXPECTED = [
    "transformer.embed.weight",
    "transformer.blocks.0.attn_norm.weight",
    "transformer.blocks.0.attn.qkv_proj.weight",
    "transformer.blocks.0.attn.o_proj.weight",
    "transformer.blocks.0.attn.q_norm.weight",
    "transformer.blocks.0.attn.k_norm.weight",
    "transformer.blocks.0.mlp_norm.weight",
    "transformer.blocks.0.mlp.gate_up_proj.weight",
    "transformer.blocks.0.mlp.down_proj.weight",
    "transformer.final_norm.weight",
]


def test_state_dict_layout_is_stable():
    assert list(build_model(CFG).state_dict()) == EXPECTED


def test_untied_adds_exactly_one_key():
    import dataclasses

    cfg = dataclasses.replace(CFG, tie_embeddings=False)
    keys = list(build_model(cfg).state_dict())
    assert keys[-1] == "transformer.lm_head.weight"
    assert len(keys) == len(EXPECTED) + 1
