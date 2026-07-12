"""FLOP accounting for throughput (tokens/s) and MFU reporting."""
from __future__ import annotations

from snapdragon.config.model import ModelConfig

# Peak dense bf16 tensor-core TFLOPs (vendor numbers, no sparsity), used only for MFU reporting.
PEAK_BF16_TFLOPS = {
    "rtx 3060": 25.0, "rtx 3070": 40.0, "rtx 3080": 59.5, "rtx 3090": 71.0,
    "rtx 4060 ti": 44.0, "rtx 4070": 58.0, "rtx 4070 ti": 80.0, "rtx 4080": 97.0, "rtx 4090": 165.0,
    "a10g": 70.0, "a100": 312.0, "l4": 121.0, "l40s": 362.0, "h100": 989.0,
}


def flops_per_token(cfg: ModelConfig, seq_len: int | None = None, recompute: str = "none") -> float:
    """Training FLOPs per token: 6 * matmul params (fwd 2, bwd 4) + attention scores/values.

    The output projection counts as matmul params even when tied (the lookup side is free).
    ``recompute``: "none" | "selective" | "full": checkpointed blocks redo their forward in backward.
    """
    T = seq_len or cfg.max_seq_len
    body = cfg.num_params(include_embeddings=False)
    matmul_params = body + cfg.vocab_size * cfg.d_model
    attn = 12 * cfg.n_layers * cfg.n_heads * cfg.head_dim * T / 2  # causal: half the T x T scores
    fwd_bwd = 6 * matmul_params + attn
    extra = {"none": 0.0, "selective": 0.5, "full": 1.0}[recompute]  # fraction of body fwd redone
    return fwd_bwd + extra * (2 * body + attn / 3)


def mfu(tokens_per_sec: float, cfg: ModelConfig, peak_tflops: float, seq_len: int | None = None) -> float:
    """Model FLOPs utilisation (recompute is deliberately not counted as useful work)."""
    return tokens_per_sec * flops_per_token(cfg, seq_len) / (peak_tflops * 1e12)


def peak_tflops_for(device_name: str) -> float | None:
    name = device_name.lower()
    best = None
    for key, val in PEAK_BF16_TFLOPS.items():
        if key in name and (best is None or len(key) > len(best[0])):
            best = (key, val)
    return best[1] if best else None
