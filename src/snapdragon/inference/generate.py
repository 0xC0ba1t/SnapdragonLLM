"""Autoregressive generation with a KV cache: one prefill pass, then one cheap step per token."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

from snapdragon.inference.kv_cache import KVCache
from snapdragon.inference.sampling import SamplingConfig, sample_next


@torch.no_grad()
def stream_generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    sampling: SamplingConfig | None = None,
    stop_ids: Iterable[int] = (),
    seed: int | None = None,
    cache_dtype: torch.dtype | None = None,
) -> Iterator[int]:
    """Yield new token ids one at a time (the stop token, if hit, is yielded last then generation ends).

    The prompt is left-truncated if prompt + new tokens would not fit the model's context window.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty (start with a BOS token)")
    sampling = sampling or SamplingConfig()
    cfg = model.cfg
    device = next(model.parameters()).device
    room = cfg.max_seq_len - max_new_tokens
    if room < 1:
        raise ValueError(f"max_new_tokens {max_new_tokens} leaves no room in a {cfg.max_seq_len}-token context")
    prompt_ids = prompt_ids[-room:]
    stops = set(stop_ids)
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    was_training = model.training
    model.eval()
    try:
        cache = KVCache(cfg, 1, len(prompt_ids) + max_new_tokens, device,
                        cache_dtype or next(model.parameters()).dtype)
        history = torch.tensor([prompt_ids], device=device)
        hidden = model.transformer(history, cache=cache)  # prefill
        for i in range(max_new_tokens):
            logits = model.transformer.logits(hidden[:, -1])
            token = sample_next(logits, sampling, generator, history)
            yield int(token)
            if int(token) in stops or i == max_new_tokens - 1:
                return  # nothing left to predict, so skip the (wasted) final forward pass
            history = torch.cat([history, token[:, None]], dim=1)
            hidden = model.transformer(token[:, None], cache=cache)  # decode one token
    finally:
        model.train(was_training)


@dataclass
class GenerationResult:
    token_ids: list[int]
    finish_reason: str  # "stop" | "length"


def generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    sampling: SamplingConfig | None = None,
    stop_ids: Iterable[int] = (),
    seed: int | None = None,
) -> GenerationResult:
    stops = set(stop_ids)
    out = list(stream_generate(model, prompt_ids, max_new_tokens, sampling, stops, seed))
    reason = "stop" if out and out[-1] in stops else "length"
    return GenerationResult(out, reason)
