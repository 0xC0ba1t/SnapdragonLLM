import pytest
import torch

from snapdragon.nn.docmask import FLEX_AVAILABLE, dense_doc_mask, flex_doc_block_mask


def test_dense_mask_is_causal_and_blocks_other_documents():
    doc = torch.tensor([[0, 0, 0, 1, 1]])
    m = dense_doc_mask(doc)[0, 0]
    assert m.shape == (5, 5)
    assert m[2, 0]  # same doc, causal
    assert m[2, 2]
    assert not m[0, 2]                  # future
    assert not m[3, 2]  # earlier document
    assert not m[4, 0]
    assert m[4, 3]


def test_batch_dimension_is_independent():
    doc = torch.tensor([[0, 0, 1], [0, 1, 1]])
    m = dense_doc_mask(doc)
    assert m.shape == (2, 1, 3, 3)
    assert m[0, 0, 1, 0]
    assert not m[1, 0, 1, 0]


@pytest.mark.skipif(not FLEX_AVAILABLE, reason="no flexattention")
def test_flex_matches_dense_attention():
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    B, H, T, D = 2, 2, 24, 8
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    doc = torch.tensor([[0] * 10 + [1] * 14, [0] * 5 + [1] * 5 + [2] * 14])
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense_doc_mask(doc))
    out = flex_attention(q, k, v, block_mask=flex_doc_block_mask(doc))
    assert torch.allclose(out, ref, atol=1e-5)
