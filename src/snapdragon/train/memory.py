"""Predict peak training memory (GB) from the configs, without allocating the model.

The estimate is deliberately simple and a little conservative; ``scripts/estimate_memory.py`` prints it
and ``train.py`` logs the *measured* peak so the two can be compared on real hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from snapdragon.config.schema import Config
from snapdragon.model.factory import build_model
from snapdragon.optim.memory import estimate_state_bytes

GB = 1e9


@dataclass
class MemoryEstimate:
    weights: float
    grads: float
    optimizer: float
    activations: float
    logits: float
    overhead: float
    ema: float = 0.0

    @property
    def total(self) -> float:
        return self.weights + self.grads + self.optimizer + self.activations + self.logits + self.overhead + self.ema

    def rows(self) -> list[tuple[str, float]]:
        return [("weights (fp32 master)", self.weights), ("gradients (fp32)", self.grads),
                ("optimizer state", self.optimizer), ("activations", self.activations),
                ("loss logits chunk", self.logits), ("workspace + cuda context", self.overhead),
                *([("weight EMA (fp32)", self.ema)] if self.ema else []), ("TOTAL", self.total)]


def activation_bytes_per_token_per_layer(cfg, mode: str) -> float:
    """bf16 activations a block keeps for backward, per token.

    Without checkpointing a block stores roughly: norm inputs (2d), qkv (q+2kv), attention output
    (d, flash keeps only O + logsumexp), o_proj input (d), gate/up outputs (2h), silu*up (h) -> about
    (6d + q + 2kv + 3h) values. Full checkpointing keeps only the block input (d); SAC keeps the
    matmul outputs but drops the pointwise intermediates (about 60% of the full set).
    """
    m = cfg
    d, h = m.d_model, m.mlp_hidden
    kv = m.n_kv_heads * m.head_dim
    full = (6 * d + m.n_heads * m.head_dim + 2 * kv + 3 * h) * 2  # bf16
    if mode == "none":
        return full
    if mode == "sac":
        return 0.6 * full
    return d * 2  # "full" and the checkpointed blocks of "selective"


def estimate_memory(cfg: Config) -> MemoryEstimate:
    model = build_model(cfg, device="meta")
    n_params = sum(p.numel() for p in model.parameters())
    m, t = cfg.model, cfg.train
    opt = estimate_state_bytes(model, cfg.optim)["total"]
    offloaded = cfg.optim.offload_states

    mode = m.act_ckpt
    if mode == "selective":
        k = m.act_ckpt_every
        per_layer = (activation_bytes_per_token_per_layer(m, "full") / k
                     + activation_bytes_per_token_per_layer(m, "none") * (1 - 1 / k))
    else:
        per_layer = activation_bytes_per_token_per_layer(m, mode)
    tokens = t.micro_batch_size * t.seq_len
    acts = per_layer * m.n_layers * tokens
    if mode in ("full", "selective"):
        acts += activation_bytes_per_token_per_layer(m, "none") * tokens  # one block recomputed at a time
    acts += tokens * m.d_model * 2 * 3  # embeddings out, final norm in/out
    # the per-token formulas above count bf16 (2 byte) activations; without autocast they are fp32
    fp32 = t.precision == "fp32"
    acts *= 2.0 if fp32 else 1.0

    chunk = min(t.ce_chunk_tokens, tokens)
    logits = chunk * m.vocab_size * (4 + 4 + (4 if fp32 else 2))  # fp32 logits + fp32 softmax grad + compute-dtype copy
    return MemoryEstimate(
        weights=n_params * 4 / GB,
        grads=n_params * 4 / GB,
        optimizer=0.0 if offloaded else opt / GB,
        activations=acts / GB,
        logits=logits / GB,
        overhead=1.0,
        ema=n_params * 4 / GB if t.ema_decay > 0 else 0.0,
    )


def measured_peak_gb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / GB if device.type == "cuda" else None
