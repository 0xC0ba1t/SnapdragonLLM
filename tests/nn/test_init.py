import math

import torch
from torch import nn

from snapdragon.config.model import ModelConfig
from snapdragon.nn.block import Block
from snapdragon.nn.init import init_weights


def _cfg(**kw):
    return ModelConfig(vocab_size=64, d_model=256, n_layers=8, n_heads=8, n_kv_heads=4, mlp_hidden=512,
                       max_seq_len=32, **kw)


def test_standard_and_residual_stds():
    torch.manual_seed(0)
    cfg = _cfg()
    block = Block(cfg)
    init_weights(block, cfg)
    assert block.attn.qkv_proj.weight.std().item() == __import__("pytest").approx(0.02, rel=0.05)
    expected_res = 0.02 / math.sqrt(2 * cfg.n_layers)
    assert block.attn.o_proj.weight.std().item() == __import__("pytest").approx(expected_res, rel=0.05)
    assert block.mlp.down_proj.weight.std().item() == __import__("pytest").approx(expected_res, rel=0.05)


def test_embedding_and_norm_defaults():
    cfg = _cfg()
    emb = nn.Embedding(64, 256)
    init_weights(emb, cfg)
    assert abs(emb.weight.std().item() - 0.02) < 0.002
    block = Block(cfg)
    init_weights(block, cfg)
    assert torch.equal(block.attn_norm.weight, torch.ones(256))
