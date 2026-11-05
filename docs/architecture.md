# Architecture

SnapdragonLLM v3 is a decoder-only transformer in the Llama/Qwen3 family, written from scratch in
PyTorch. The 500M preset (`configs/model/snapdragon-500m.yaml`) has **528M parameters**.

| | |
|---|---|
| layers / width | 28 / 1280 |
| attention | 20 query heads, 5 key/value heads (grouped-query, 4:1), head dim 64 |
| MLP | SwiGLU, hidden 3456 (a multiple of 128 for tensor cores) |
| norm | RMSNorm, pre-norm, plus RMSNorm on q and k per head (QK-norm) |
| positions | RoPE, theta 500,000, half-split convention |
| embeddings | tied input/output, vocabulary 32,768 |
| context | 2048 |
| biases | none |

## One block

```
x ──► RMSNorm ─► QKV proj ─► q,k: RMSNorm(head) ─► RoPE ─► SDPA (causal, GQA) ─► O proj ─┐
 └───────────────────────────────────────────────────────────────────────────────────────(+)─► y
y ──► RMSNorm ─► gate/up proj (fused) ─► silu(gate) * up ─► down proj ─────────────────────┐
 └───────────────────────────────────────────────────────────────────────────────────────(+)─► out
```

Why these choices:

* **GQA (4:1)** cuts the KV cache 4x and the K/V projection cost, with negligible quality loss.
* **QK-norm** bounds attention logits, which is what lets Muon's large learning rate train stably.
* **Pre-norm + depth-scaled init**: output projections are initialised with `std / sqrt(2 * n_layers)`
  so the residual stream's variance does not grow with depth.
* **Tied embeddings** save 42M parameters (8% of the model) at this scale.
* **No biases**: fewer parameters to shard, decay or quantize, and no measurable loss.

## The loss never builds the logits

A (batch x 2048 x 32768) logits tensor is 0.5 GB per sequence in fp32 and is the single largest
activation in a small model. `nn/functional/ce.py` implements a custom autograd function that fuses the
output projection with the cross-entropy: it processes `ce_chunk_tokens` tokens at a time, saves only
the per-token logsumexp, and recomputes each chunk in backward. A unit test asserts that nothing the
size of the full logits is ever saved for backward. A small z-loss (`1e-4 * logsumexp^2`) keeps the
logit scale from drifting.

## Optimizers

Hidden weight matrices are trained by **Muon** (momentum, then an approximate orthogonalisation of
the update by five Newton-Schulz iterations, in bf16, batched across same-shaped layers). It keeps
one bf16 buffer per parameter. Embeddings and gains use **8-bit AdamW** with blockwise absmax
quantization (2 bytes of state per parameter instead of 8). Together the optimizer state of the 500M
model is about **1.1 GB instead of 4.2 GB**. See [training-on-a-16gb-gpu.md](training-on-a-16gb-gpu.md).

## Checkpoints and compatibility

* `state_dict` keys are pinned by a test, so a refactor cannot silently orphan saved checkpoints.
* Training checkpoints are directories written atomically (temp dir, `COMPLETE` marker, rename) and
  hold weights, optimizer, schedule, data position, RNG and counters. Resume is bit-for-bit on CPU
  (tested).
* `scripts/export_model.py` writes `safetensors` and a **Qwen3-format** Hugging Face export. Our
  block is architecturally a Qwen3 block, and a test checks logit and greedy-generation parity with
  `transformers`.

## Package map

```
config/      typed dataclass configs, YAML loader with unknown-key suggestions, dotted overrides
tokenizer/   reference BPE (readable, test oracle) + fast tokenizer for real runs, chat template
data/        shards, manifest, memmap dataset, packing, mixture, resumable sampler, loaders
nn/          RMSNorm, RoPE, attention, SwiGLU, block, init, fused chunked loss, doc masks
model/       Transformer, CausalLM, factory (meta device), compile, FLOPs, summary
optim/       Muon, AdamW8bit, quantization, hybrid facade, schedules, clipping, memory
train/       Trainer, callbacks, checkpointing, curriculum, EMA, memory estimator, benchmark
distributed/ torchrun env, DDP wrapper (FSDP2 is experimental)
eval/        likelihood multiple-choice, HellaSwag/ARC/PIQA/LAMBADA, perplexity
inference/   KV cache, sampling, streaming generation, chat
sft/         chat data, assistant-only masking, packing, loader
export/      safetensors, Hugging Face (Qwen3)
```
