"""Document-boundary attention masks for packed sequences (causal AND same document)."""
from __future__ import annotations

import torch

try:  # FlexAttention needs torch >= 2.5
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_AVAILABLE = True
except ImportError:  # pragma: no cover
    BlockMask = None
    create_block_mask = None
    FLEX_AVAILABLE = False


def dense_doc_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """(B, T) document ids -> (B, 1, T, T) bool mask, True where attention is allowed.

    Memory is O(B T^2), fine for tests and short sequences; use the flex block mask for training.
    """
    T = doc_ids.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
    same = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    return (causal & same).unsqueeze(1)


def flex_doc_block_mask(doc_ids: torch.Tensor):
    """Sparse FlexAttention mask: whole blocks of cross-document attention are skipped."""
    if not FLEX_AVAILABLE:
        raise RuntimeError("FlexAttention is not available in this PyTorch build")
    B, T = doc_ids.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    return create_block_mask(mask_mod, B, None, T, T, device=doc_ids.device)
