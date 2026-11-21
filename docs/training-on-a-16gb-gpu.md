# Training the 500M model on one consumer GPU

The goal of v3's training stack is that **one 8-16 GB GPU is enough**, not that it is fast. Pretraining
a 528M model on 10B tokens is ~4e19 FLOPs; on a single consumer card that is weeks, and this page is
honest about it. Everything below about *memory* is predicted by `scripts/estimate_memory.py` and
checked by `tests/gpu/test_memory_budget.py` on a real GPU; everything about *time* is a FLOP-model
estimate until you run `scripts/benchmark.py` on your own card.

## Where the memory goes (500M, 2048 context)

| component | how it is kept small | 16 GB preset |
|---|---|---|
| fp32 master weights | needed for stable updates | 2.1 GB |
| fp32 gradients | accumulated across micro-batches | 2.1 GB |
| optimizer state | Muon bf16 momentum + 8-bit AdamW | **1.1 GB** (fp32 AdamW would be 4.2 GB) |
| activations | bf16 autocast, selective checkpointing, flash attention | ~5.6 GB |
| loss logits | fused chunked cross-entropy (2048-token chunks) | 0.7 GB |
| workspace | CUDA context, allocator | ~1 GB |
| **total (predicted)** | | **~12.5 GB** |

Presets (all train at the same global batch of 524,288 tokens, so results are comparable):

| preset | GPU | micro-batch | checkpointing | optimizer state | predicted peak |
|---|---|---|---|---|---|
| `pretrain-16gb` | 16 GB | 4 | `sac` | on GPU | ~12.5 GB |
| `pretrain-12gb` | 12 GB | 2 | `full` | on GPU | ~7.4 GB |
| `pretrain-8gb` | 8 GB | 1 | `full` | **in host RAM** | ~6 GB |

Run `python scripts/estimate_memory.py --config configs/train/pretrain-16gb.yaml` to see the table for
your overrides. The estimator is deliberately conservative.

## The techniques, in the order they matter

1. **Fused chunked loss** - never materialises the logits (the biggest single activation).
2. **8-bit optimizer state + Muon** - 4x less optimizer memory than fp32 AdamW.
3. **Activation checkpointing** - `sac` keeps matmul outputs and recomputes cheap pointwise ops (no
   matmul is redone); `full` recomputes each block (+33% FLOPs) for the smallest footprint; `selective`
   checkpoints every k-th block.
4. **bf16 autocast** with fp32 master weights; flash/mem-efficient attention through PyTorch SDPA.
5. **Gradient accumulation** to the 0.5M-token batch regardless of micro-batch.
6. **Host offload of optimizer state** (`optim.offload_states: true`) for 8 GB cards: ~1 GB moves over
   PCIe once per optimizer step, which is negligible against a multi-second step.
7. `expandable_segments` allocator + one host sync per step (see `utils/device.py`, `train/trainer.py`).
8. **Sequence-length warmup** (512 to 2048 over the first 500M tokens): shorter early contexts are more
   stable at high learning rates. Tokens per step stay constant.

## Smaller presets for quick runs

Before committing weeks to the 500M model, run a small one on your real data. Each is trained for about
20 tokens per parameter with the same recipe (Muon + 8-bit AdamW, WSD schedule, sequence-length warmup).

| preset | parameters | tokens | predicted memory | rough time on a 4060 Ti-class / 4090 |
|---|---|---|---|---|
| `pretrain-25m` | 25M | 0.5B | ~5 GB | 2-4 h / 0.5-1 h |
| `pretrain-50m` | 52M | 1B | ~5.5 GB | 8-16 h / 2-4 h |
| `pretrain-125m` | 126M | 10B | n/a (micro-batch 8) | 2-8 days |

Times are FLOP-model estimates at 15-30% utilisation (small models use a GPU less efficiently); they are not
measurements. The 25M preset is the fastest way to confirm that tokenizer, data, training, evaluation and
generation all work on your hardware.

## How accurate is the estimate?

One real data point so far (Colab T4, 500M, `pretrain-16gb` shapes, bf16, `sac`, no compile): predicted
**12.5 GB**, measured peak **13.5 GB**, about 8% higher. A T4 has no fused attention kernel, which adds
transient memory that Ampere or newer cards should not need, but that is not yet measured. Plan for the
estimate being 10% optimistic and leave headroom: on a 16 GB card (about 14.5 GB usable) the default
preset is tight, and `model.act_ckpt=full` or `train.micro_batch_size=2` is the safe fallback.

`train.precision=fp32` is a debugging mode: activations are twice as large and the estimator now says so
(the 16 GB preset no longer fits in fp32, which matches what happened on the T4).

## How long will it take?

Training FLOPs are `3.61 GFLOPs/token` (`sac`) or `4.73` (`full`, with recompute) at 2048 context.
Assuming 25-35% model FLOPs utilisation, **10B tokens** takes roughly:

| GPU (bf16 peak) | `sac`, 25-35% MFU | `full`, 25-35% MFU |
|---|---|---|
| RTX 4060 Ti 16GB (44 TFLOPs) | 27-38 days | 36-50 days |
| RTX 4070 (58) | 21-29 days | 27-38 days |
| RTX 3090 (71) | 17-24 days | 22-31 days |
| RTX 4080 (97) | 12-17 days | 16-23 days |
| RTX 4090 (165) | 7-10 days | 10-13 days |

The 126M preset (`pretrain-125m`) needs about a quarter of that: roughly 2-8 days on the same cards.

**Plan around the schedule, not the clock.** The default LR schedule is warmup-stable-decay (WSD): the
learning rate stays flat, then decays over the last 20% of steps. You can stop at *any* checkpoint,
set `train.max_tokens` to what you have already trained plus a short decay window, and resume to run
just the decay. A 2-3B-token run of the 500M model is a sensible first milestone.

Measure your own card in a minute:

    python scripts/benchmark.py --config configs/train/pretrain-16gb.yaml --steps 3
    python scripts/benchmark.py --config configs/train/pretrain-16gb.yaml model.act_ckpt=full train.micro_batch_size=2

It prints predicted memory, measured tokens/s, MFU, peak memory, and the projected days.

## Which GPUs work well

Ampere or newer (RTX 30/40, A10/A100, L4) is the intended target: native bf16 and the fused flash
attention kernel. **Turing/Volta cards (T4, V100, RTX 20xx) run the code but are not a good fit**:

* no hardware bf16, so autocast is emulated or skipped and `torch.compile` skips bf16 graphs;
* no flash attention, and PyTorch's grouped-query SDPA falls back to the math path, which materialises
  the (T x T) attention scores;
* measured on a Colab T4 (16 GB), 500M, fp32: about 0.6k tokens/s (roughly 190 days for 10B tokens),
  peak 9.1 GB at micro-batch 4. Treat a T4 as a correctness and smoke-test machine only.

If you do hit out-of-memory on such a card, use `model.act_ckpt=full` and a smaller
`train.micro_batch_size`. (An earlier `sac` policy also kept the math-path score matrices and ran a T4
out of memory in bf16; that is fixed, but `full` remains the safe choice off Ampere.)

## Practical notes

* `act_ckpt: sac` (the 16 GB preset) runs **without** `torch.compile`: selective checkpointing and compiled
  blocks are incompatible (the backward recompute fails), so compile is skipped with a warning. Use
  `model.act_ckpt=full` or `selective` if you want compile; benchmark both on your GPU.
* Use Linux or WSL2 for training: `torch.compile` and Triton work there. On native Windows set
  `train.compile=false`; it still runs, just slower.
* Re-running the same command resumes from the newest checkpoint. Ctrl-C (or SIGTERM) writes a final
  checkpoint first.
* `train.loss_spike_factor: 3.0` skips steps whose loss jumps above 3x the running average, and ten
  skipped steps in a row aborts the run instead of burning compute on a diverged model.
* Expect `data/` to hold ~20 GB of tokenized shards for 10B tokens (uint16 = 2 bytes per token).
