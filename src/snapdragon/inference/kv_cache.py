"""A static, preallocated key/value cache (no per-token allocation, friendly to torch.compile)."""
from __future__ import annotations

import torch

from snapdragon.config.model import ModelConfig


class KVCache:
    """Keys/values for every layer in two tensors of shape (n_layers, B, n_kv_heads, max_len, head_dim).

    ``pos`` is how many positions are filled (identical for all layers). ``update`` writes the new
    keys/values at ``[pos, pos + T)`` and returns views over ``[0, pos + T)``; the caller advances
    ``pos`` once per forward pass via ``advance``.
    """

    def __init__(
        self, cfg: ModelConfig, batch_size: int, max_len: int | None = None,
        device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_len = max_len or cfg.max_seq_len
        if self.max_len > cfg.max_seq_len:
            raise ValueError(f"cache length {self.max_len} exceeds the model's max_seq_len {cfg.max_seq_len}")
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, self.max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = k_new.shape[2]
        if self.pos + T > self.max_len:
            raise ValueError(f"KV cache overflow: {self.pos} + {T} > {self.max_len}")
        end = self.pos + T
        self.k[layer, :, :, self.pos : end] = k_new.to(self.k.dtype)
        self.v[layer, :, :, self.pos : end] = v_new.to(self.v.dtype)
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n: int) -> None:
        self.pos += n

    def reset(self) -> None:
        self.pos = 0

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2
