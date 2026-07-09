"""CausalLM: the transformer plus the (fused, chunked) language-modelling loss."""
from __future__ import annotations

import torch
from torch import nn

from snapdragon.config.model import ModelConfig
from snapdragon.model.transformer import Transformer
from snapdragon.nn.lm_loss import lm_loss


class CausalLM(nn.Module):
    """``forward(tokens, targets)`` returns ``(loss, ce, z)`` (everything DDP needs goes through
    forward); without targets it returns full logits, meant for short inputs (eval, generation)."""

    def __init__(self, cfg: ModelConfig, z_loss_coeff: float = 1e-4, ce_chunk_tokens: int = 4096) -> None:
        super().__init__()
        self.cfg = cfg
        self.z_loss_coeff = z_loss_coeff
        self.ce_chunk_tokens = ce_chunk_tokens
        self.transformer = Transformer(cfg)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attn_mask=None,
    ):
        hidden = self.transformer(tokens, position_ids, attn_mask)
        if targets is None:
            return self.transformer.logits(hidden)
        out = lm_loss(
            hidden, self.transformer.output_weight, targets, self.z_loss_coeff, self.ce_chunk_tokens
        )
        return out.loss, out.ce, out.z
