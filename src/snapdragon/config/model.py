"""Model architecture configuration."""
from __future__ import annotations

from dataclasses import dataclass

ACT_CKPT_MODES = ("none", "selective", "full")


def suggest_mlp_hidden(d_model: int, ratio: float = 8 / 3, multiple: int = 128) -> int:
    """SwiGLU hidden size: ~ratio * d_model (8/3 keeps params equal to a 4x GELU MLP), rounded up
    to a multiple of ``multiple`` so matmul dimensions stay tensor-core friendly."""
    raw = ratio * d_model
    return int(-(-raw // multiple) * multiple)
ATTN_IMPLS = ("sdpa", "flex")


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 1280
    n_layers: int = 28
    n_heads: int = 20
    n_kv_heads: int = 5
    head_dim: int = 0  # 0 means d_model // n_heads
    mlp_hidden: int = 3456
    max_seq_len: int = 2048
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True
    init_std: float = 0.02
    attn_impl: str = "sdpa"
    act_ckpt: str = "none"
    act_ckpt_every: int = 1  # for "selective": checkpoint every k-th block

    def __post_init__(self) -> None:
        if self.head_dim == 0:
            if self.d_model % self.n_heads:
                raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
            self.head_dim = self.d_model // self.n_heads
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads {self.n_heads} not divisible by n_kv_heads {self.n_kv_heads}")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.act_ckpt not in ACT_CKPT_MODES:
            raise ValueError(f"act_ckpt must be one of {ACT_CKPT_MODES}, got {self.act_ckpt!r}")
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {self.attn_impl!r}")
        if self.act_ckpt_every < 1:
            raise ValueError("act_ckpt_every must be >= 1")

    @property
    def kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def num_params(self, include_embeddings: bool = True) -> int:
        """Analytic parameter count (matches the real module; tested against it later)."""
        d, hd = self.d_model, self.head_dim
        attn = d * self.n_heads * hd + 2 * d * self.n_kv_heads * hd + self.n_heads * hd * d
        if self.qk_norm:
            attn += 2 * hd
        mlp = 3 * d * self.mlp_hidden
        per_layer = attn + mlp + 2 * d
        total = self.n_layers * per_layer + d  # + final norm
        if include_embeddings:
            emb = self.vocab_size * d
            total += emb if self.tie_embeddings else 2 * emb
        return total
