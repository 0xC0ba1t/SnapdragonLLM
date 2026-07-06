import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.nn.attention import Attention
from snapdragon.nn.rope import RotaryEmbedding


@pytest.fixture
def cfg():
    return ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=32)


def _rope(cfg):
    return RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)


def test_output_shape(cfg):
    attn = Attention(cfg)
    assert attn(torch.randn(2, 10, 32), _rope(cfg)).shape == (2, 10, 32)


def test_causality_future_tokens_do_not_change_past_outputs(cfg):
    attn = Attention(cfg).eval()
    rope = _rope(cfg)
    x = torch.randn(1, 12, 32)
    y1 = attn(x, rope)
    x2 = x.clone()
    x2[:, 8:] = torch.randn(1, 4, 32)
    y2 = attn(x2, rope)
    assert torch.allclose(y1[:, :8], y2[:, :8], atol=1e-5)
    assert not torch.allclose(y1[:, 8:], y2[:, 8:], atol=1e-5)


def test_param_count_matches_config_formula(cfg):
    attn = Attention(cfg)
    n = sum(p.numel() for p in attn.parameters())
    d, hd = cfg.d_model, cfg.head_dim
    expected = d * cfg.n_heads * hd + 2 * d * cfg.n_kv_heads * hd + cfg.n_heads * hd * d + 2 * hd
    assert n == expected


def test_gqa_equals_mha_with_repeated_kv(cfg):
    """Grouped heads must behave like full MHA whose kv heads are copies."""
    attn = Attention(cfg).eval()
    rope = _rope(cfg)
    x = torch.randn(1, 9, 32)
    out = attn(x, rope)

    B, T, _ = x.shape
    q, k, v = attn.qkv_proj(x).split(attn._split, dim=-1)
    q = q.view(B, T, 4, 8).transpose(1, 2)
    k = k.view(B, T, 2, 8).transpose(1, 2).repeat_interleave(2, dim=1)
    v = v.view(B, T, 2, 8).transpose(1, 2).repeat_interleave(2, dim=1)
    q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = rope(q, k)
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.o_proj(y.transpose(1, 2).reshape(B, T, -1))
    assert torch.allclose(out, ref, atol=1e-5)


def test_no_qk_norm_variant(cfg):
    cfg.qk_norm = False
    attn = Attention(cfg)
    assert isinstance(attn.q_norm, torch.nn.Identity)
    assert attn(torch.randn(1, 4, 32), _rope(cfg)).shape == (1, 4, 32)


def test_backward_runs(cfg):
    attn = Attention(cfg)
    attn(torch.randn(2, 6, 32), _rope(cfg)).sum().backward()
    assert all(p.grad is not None for p in attn.parameters())


def test_document_mask_isolates_documents(cfg):
    from snapdragon.data.packing import doc_ids_from_tokens, position_ids_from_tokens
    from snapdragon.nn.docmask import dense_doc_mask

    attn = Attention(cfg).eval()
    rope = _rope(cfg)
    eos = 63
    tokens = torch.tensor([[1, 2, 3, eos, 4, 5, 6, 7]])
    pos = position_ids_from_tokens(tokens, eos)
    mask = dense_doc_mask(doc_ids_from_tokens(tokens, eos))
    x = torch.randn(1, 8, 32)
    y1 = attn(x, rope, pos, mask)
    x2 = x.clone()
    x2[:, :4] = torch.randn(1, 4, 32)  # change the whole first document
    y2 = attn(x2, rope, pos, mask)
    assert torch.allclose(y1[:, 4:], y2[:, 4:], atol=1e-5)  # second document unaffected
    assert not torch.allclose(y1[:, :4], y2[:, :4], atol=1e-5)


def test_dense_causal_mask_equals_default_path(cfg):
    from snapdragon.nn.docmask import dense_doc_mask

    attn = Attention(cfg).eval()
    rope = _rope(cfg)
    x = torch.randn(1, 8, 32)
    one_doc = dense_doc_mask(torch.zeros(1, 8, dtype=torch.long))
    assert torch.allclose(attn(x, rope), attn(x, rope, None, one_doc), atol=1e-5)


def test_flex_block_mask_path_matches_dense_mask(cfg):
    from snapdragon.nn.docmask import FLEX_AVAILABLE, dense_doc_mask, flex_doc_block_mask

    if not FLEX_AVAILABLE:
        pytest.skip("no flexattention")
    attn = Attention(cfg).eval()
    rope = _rope(cfg)
    doc = torch.tensor([[0] * 5 + [1] * 11])
    x = torch.randn(1, 16, 32)
    dense = attn(x, rope, None, dense_doc_mask(doc))
    flex = attn(x, rope, None, flex_doc_block_mask(doc))
    assert torch.allclose(dense, flex, atol=1e-5)
