# Multi-GPU training

The target is one consumer GPU, but the trainer is multi-GPU ready via **DDP** (data parallel).

    torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/train/pretrain-16gb.yaml

* `train.py` reads `RANK` / `LOCAL_RANK` / `WORLD_SIZE`, initialises NCCL (GLOO on CPU), pins each rank to
  its GPU and wraps the model in `DistributedDataParallel`. Single-process runs create no process group.
* The global batch (`train.global_batch_tokens`) is fixed; gradient accumulation per rank shrinks with the
  number of ranks (`grad_accum_steps = global_batch / (micro_batch * seq_len * world_size)`), so the
  optimization is identical on 1 or N GPUs. Non-final micro-batches use `no_sync()` so gradients are
  all-reduced once per step.
* Ranks read disjoint slices of each global batch from the same deterministic stream. Changing the number
  of GPUs mid-run does not repeat or skip data, so a run can resume on a different GPU count.
* Only rank 0 logs, evaluates benchmarks and writes checkpoints; a barrier after each checkpoint keeps the
  other ranks from racing ahead into a collective timeout. Metrics (loss, validation) are averaged across
  ranks.
* Each rank holds a full copy of weights, gradients and optimizer state, so per-GPU memory is the same as
  single GPU (see [training-on-a-16gb-gpu.md](training-on-a-16gb-gpu.md)). DDP buys speed, not capacity.

## FSDP2 (experimental)

`snapdragon.distributed.fsdp` shards parameters, gradients and optimizer state with `fully_shard`, which
would be the way to train something larger than one GPU can hold. It is **not selectable from `train.py`**:
Muon needs whole matrices (it falls back to stock AdamW under FSDP) and checkpointing sharded weights needs
a collective full-state-dict gather that the trainer does not perform yet. It is kept as tested library
code (its two-process test runs where gloo is available).

## Testing without GPUs

`snapdragon.distributed.testing.run_distributed(fn, world_size)` spawns gloo CPU processes and returns each
rank's result; `tests/distributed` uses it to check that DDP gradients equal the single-process full-batch
gradients. The Windows PyTorch wheels lack a usable gloo transport, so those tests skip there and run on
Linux CI.
