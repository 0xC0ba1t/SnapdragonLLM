import torch

from snapdragon.config.model import ModelConfig
from snapdragon.nn.block import Block
from snapdragon.nn.rope import RotaryEmbedding


def _cfg():
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=32)


def test_residual_shape_and_param_count():
    cfg = _cfg()
    block = Block(cfg)
    rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
    assert block(torch.randn(2, 7, 32), rope).shape == (2, 7, 32)
    n = sum(p.numel() for p in block.parameters())
    assert n == (cfg.num_params(include_embeddings=False) - cfg.d_model) // cfg.n_layers


def test_zeroed_branches_make_block_the_identity():
    cfg = _cfg()
    block = Block(cfg)
    torch.nn.init.zeros_(block.attn.o_proj.weight)
    torch.nn.init.zeros_(block.mlp.down_proj.weight)
    x = torch.randn(1, 5, 32)
    assert torch.equal(block(x, RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)), x)
