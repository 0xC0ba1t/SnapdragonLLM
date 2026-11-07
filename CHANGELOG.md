# Changelog

## 0.3.0 (v3, in development)

A complete rewrite. v2's single-file GPT-2 124M reproduction moved to `legacy/v2/`.

### Model
- 528M-parameter decoder: grouped-query attention, QK-norm, RoPE, SwiGLU, RMSNorm, tied embeddings.
- Meta-device construction (`device="meta"`) and deterministic, device-independent weight init.
- Selective (`sac`), per-block (`selective`) and full activation checkpointing; regional `torch.compile`.
- KV-cache incremental decoding; document attention masks (dense and FlexAttention).

### Memory and speed
- Fused, chunked linear + cross-entropy + z-loss that never materialises the logits.
- Muon (bf16 momentum, batched Newton-Schulz) and a from-scratch blockwise 8-bit AdamW.
- Optional host offload of optimizer state; `expandable_segments` allocator; one host sync per step.
- `estimate_memory`, `benchmark` and `profile_train` scripts.

### Data and tokenizer
- 32k byte-level BPE (reference implementation and fast implementation sharing one contract).
- uint16 shards + manifest, quality filters, exact dedup, memmap datasets, document-aware packing.
- Deterministic mixture sampler whose only state is a global sample index (rank-count independent).

### Training
- Callback-based trainer: WSD/cosine/linear schedules, sequence-length warmup, gradient accumulation,
  loss-spike skipping, divergence guard, weight EMA, throughput/MFU, TensorBoard/W&B.
- Atomic checkpoints with bit-exact resume; graceful SIGINT/SIGTERM; DDP (FSDP2 experimental).
- SFT with assistant-only loss, packing, and `init_from` weight loading.

### Evaluation, inference, export
- HellaSwag, ARC, PIQA, LAMBADA, perplexity/bits-per-byte; in-training validation and benchmarks.
- Streaming generation (top-k/top-p/min-p/repetition penalty, stop strings), chat sessions.
- safetensors and Qwen3-format Hugging Face export with logit/generation parity tests.

### Known limitations
- Weights are not trained yet; the model card has no results.
- FSDP2 is not wired into `train.py` (needs collective checkpointing).
- CUDA, `torch.compile` and multi-process tests are written but need suitable hardware to run.
