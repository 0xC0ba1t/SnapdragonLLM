"""The decoder-only transformer."""
from __future__ import annotations

import functools

import torch
from torch import nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from snapdragon.config.model import ModelConfig
from snapdragon.nn.block import Block
from snapdragon.nn.init import init_weights
from snapdragon.nn.norm import RMSNorm
from snapdragon.nn.rope import RotaryEmbedding


def _save_ops() -> set:
    """Expensive ops whose outputs SAC keeps instead of recomputing (matmuls and attention kernels)."""
    aten = torch.ops.aten
    names = ["mm", "addmm", "bmm", "_scaled_dot_product_flash_attention",
             "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention",
             "_scaled_dot_product_flash_attention_for_cpu"]
    return {getattr(aten, n).default for n in names if hasattr(aten, n)}


_SAVE = _save_ops()


def _sac_policy(ctx, op, *args, **kwargs):
    return CheckpointPolicy.MUST_SAVE if op in _SAVE else CheckpointPolicy.PREFER_RECOMPUTE


_SAC_CONTEXT = functools.partial(create_selective_checkpoint_contexts, _sac_policy)


class Transformer(nn.Module):
    """Token embedding -> N pre-norm blocks -> final RMSNorm. Returns final hidden states;
    the (tied) output matrix is applied by the loss head or ``logits`` so the full
    (B, T, vocab) logits are never created during training."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """(Re)initialise every weight. Rotary tables are buffers, handled by ``rope.rebuild``."""
        init_weights(self, self.cfg)

    @property
    def output_weight(self) -> torch.Tensor:
        """The (vocab, d_model) matrix mapping hidden states to logits."""
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    def forward(
        self, tokens: torch.Tensor, position_ids: torch.Tensor | None = None, attn_mask=None, cache=None
    ) -> torch.Tensor:
        """``cache`` (a KVCache) switches to incremental decoding: tokens are the *new* tokens, their
        positions continue from ``cache.pos``, and the cache is advanced once all layers have run."""
        if tokens.shape[1] > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq_len {self.cfg.max_seq_len}")
        if cache is not None and position_ids is None:
            T = tokens.shape[1]
            position_ids = torch.arange(cache.pos, cache.pos + T, device=tokens.device).expand(tokens.shape[0], T)
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            if cache is not None:
                x = block(x, self.rope, position_ids, attn_mask, cache, i)
            elif self.training and self.cfg.act_ckpt == "sac":
                # keep matmul/attention outputs, recompute only the cheap pointwise ops
                x = checkpoint(block, x, self.rope, position_ids, attn_mask,
                               use_reentrant=False, context_fn=_SAC_CONTEXT)
            elif self.training and self._checkpoint_block(i):
                x = checkpoint(block, x, self.rope, position_ids, attn_mask, use_reentrant=False)
            else:
                x = block(x, self.rope, position_ids, attn_mask)
        if cache is not None:
            cache.advance(tokens.shape[1])
        return self.final_norm(x)

    def _checkpoint_block(self, i: int) -> bool:
        """Activation checkpointing trades compute for memory: a checkpointed block keeps only its
        input and redoes its forward during backward. "selective" does this for every k-th block."""
        mode = self.cfg.act_ckpt
        if mode == "full":
            return True
        if mode == "selective":
            return i % self.cfg.act_ckpt_every == 0
        return False

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full logits for a (small) hidden tensor, e.g. the last position during generation."""
        return hidden @ self.output_weight.to(hidden.dtype).T
