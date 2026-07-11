"""The decoder-only transformer."""
from __future__ import annotations

import torch
from torch import nn

from snapdragon.config.model import ModelConfig
from snapdragon.nn.block import Block
from snapdragon.nn.init import init_weights
from snapdragon.nn.norm import RMSNorm
from snapdragon.nn.rope import RotaryEmbedding


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
        self, tokens: torch.Tensor, position_ids: torch.Tensor | None = None, attn_mask=None
    ) -> torch.Tensor:
        if tokens.shape[1] > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, self.rope, position_ids, attn_mask)
        return self.final_norm(x)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full logits for a (small) hidden tensor, e.g. the last position during generation."""
        return hidden @ self.output_weight.to(hidden.dtype).T
