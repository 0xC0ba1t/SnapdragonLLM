# SnapdragonLLM

A **528M-parameter decoder-only language model, written from scratch in PyTorch** and engineered so that
the whole thing (data, tokenizer, pretraining, fine-tuning, evaluation, generation, export) works on **one
8-16 GB consumer GPU**.

> v3 is a ground-up rewrite. The original GPT-2 124M reproduction lives in [`legacy/v2/`](legacy/v2).
> OLMo-core was an inspiration for what a production training stack contains; no code was copied.

**Status:** code complete and CPU-tested end to end; the 500M weights have not been trained yet
(see [model card](docs/model-card.md)). Training time is weeks on a single consumer GPU, see
[the honest numbers](docs/training-on-a-16gb-gpu.md).

## What is in here

| | |
|---|---|
| **Model** | 28 layers, d=1280, grouped-query attention (20/5 heads), QK-norm, RoPE, SwiGLU, RMSNorm, tied embeddings, 32k vocab, 2048 context |
| **Fused loss** | custom autograd function: output projection + cross-entropy + z-loss in chunks, never builds the logits |
| **Optimizers** | Muon for matrices (bf16 momentum), from-scratch blockwise 8-bit AdamW for the rest: 1.1 GB of optimizer state instead of 4.2 GB |
| **Memory tools** | selective activation checkpointing, optional host offload of optimizer state, a memory estimator, a profiler |
| **Data** | custom BPE tokenizer, uint16 shards, memmap, document-aware packing, deterministic resumable sampler, prefetch |
| **Training** | WSD schedule, sequence-length warmup, loss-spike skipping, EMA, atomic bit-exact resumable checkpoints, DDP |
| **Eval** | HellaSwag, ARC, PIQA, LAMBADA, perplexity, in-training validation |
| **Inference** | KV cache, top-k/top-p/min-p/repetition penalty, streaming, chat |
| **SFT** | chat templates, assistant-only loss, packing |
| **Export** | safetensors and a Qwen3-format Hugging Face checkpoint (logit parity tested against `transformers`) |

## Quickstart

```bash
pip install -e ".[data,dev]"           # add ",logging" for TensorBoard / W&B
pytest -q -m "not gpu and not slow"    # CPU test suite (a few minutes)

# 1. tokenizer + data (FineWeb-Edu, 10B tokens ~ 20 GB on disk)
python scripts/train_tokenizer.py --hf HuggingFaceFW/fineweb-edu:sample-10BT --max-gb 2
python scripts/prepare_data.py --spec hf:HuggingFaceFW/fineweb-edu:sample-10BT \
    --tokenizer data/tokenizer/snapdragon-32k.json --out data/fineweb-edu

# 2. see what your GPU can do before committing weeks to it
python scripts/estimate_memory.py --config configs/train/pretrain-16gb.yaml
python scripts/benchmark.py       --config configs/train/pretrain-16gb.yaml --steps 3

# 3. pretrain (re-run the same command to resume; Ctrl-C saves a checkpoint first)
python scripts/train.py --config configs/train/pretrain-16gb.yaml     # or -12gb / -8gb / -125m

# 4. evaluate, generate, fine-tune, export
python scripts/eval.py --checkpoint runs/snapdragon-500m-16gb/checkpoints \
    --tokenizer data/tokenizer/snapdragon-32k.json --tasks hellaswag arc_easy piqa lambada
python scripts/generate.py --checkpoint runs/snapdragon-500m-16gb/checkpoints \
    --tokenizer data/tokenizer/snapdragon-32k.json --prompt "The theory of evolution"
python scripts/export_model.py --checkpoint runs/snapdragon-500m-16gb/checkpoints --out release --format safetensors hf
```

Smoke-test the whole pipeline on CPU in about a minute: `pytest tests/test_smoke_e2e.py -m slow`.

## Scripts

| script | purpose |
|---|---|
| `train_tokenizer.py`, `prepare_data.py`, `inspect_data.py` | build, verify and summarise the tokenizer and token shards |
| `estimate_memory.py`, `benchmark.py`, `profile_train.py` | predict, measure and profile memory and throughput |
| `train.py` | pretrain or SFT from a YAML config (resumes automatically; `torchrun` for DDP) |
| `eval.py`, `inspect_model.py` | benchmarks and perplexity; parameter breakdown and checkpoint state |
| `generate.py`, `chat.py` | text generation and interactive chat |
| `prepare_sft.py`, `export_model.py` | chat-data normalisation; safetensors / Hugging Face export |
| `gen_config_docs.py` | regenerate `docs/configuration.md` |

## Documentation

* [Architecture](docs/architecture.md) - the model, the fused loss, the optimizers, the package map
* [Training on one GPU](docs/training-on-a-16gb-gpu.md) - memory budget, presets, realistic timings
* [Multi-GPU](docs/distributed.md) - DDP, elastic resume, the experimental FSDP2 path
* [Data](docs/data.md) and [Tokenizer](docs/tokenizer.md)
* [Evaluation](docs/evaluation.md), [Inference and export](docs/inference.md), [SFT](docs/sft.md)
* [Configuration reference](docs/configuration.md) (generated from the code)
* [Model card](docs/model-card.md), [Contributing](CONTRIBUTING.md), [Changelog](CHANGELOG.md)

## Verification status (read before trusting a claim)

* **Verified by tests on CPU:** model correctness (causality, GQA, RoPE, cached decoding), exact agreement
  with `transformers` Qwen3 (logits and greedy generation), numerical equivalence of the fused loss,
  checkpoint resume bit-for-bit, optimizer state round-trips, the data pipeline, and a full
  tokenizer-to-checkpoint-to-resume smoke run.
* **Written but only runnable on the right hardware:** the CUDA tests (`pytest -m gpu`: numerics and the
  memory budget of each preset), `torch.compile` (Linux), and the multi-process DDP/FSDP tests (need a
  working gloo/nccl build; they are skipped otherwise). FSDP2 is experimental and not selectable from
  `train.py`.
* **Estimated, not measured:** GPU memory (`estimate_memory.py`) and training time (FLOP model). Use
  `scripts/benchmark.py` on your card for real numbers.

## License

AGPL-3.0, see [`LICENSE`](LICENSE).
