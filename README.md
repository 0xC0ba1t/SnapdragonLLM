# SnapdragonLLM

A ~500M parameter decoder-only language model, written from scratch in PyTorch and
engineered to be trainable on a **single 12-16 GB consumer GPU**.

> v3 is a ground-up rewrite. The original GPT-2 124M reproduction lives in `legacy/v2/`.

Status: work in progress. See `docs/` as it appears.

## Layout

- `src/snapdragon/` the library (config, tokenizer, data, nn, model, optim, train, eval, inference)
- `scripts/` command line entry points
- `configs/` YAML presets for models and training runs
- `tests/` CPU-runnable unit tests (GPU tests are marked `gpu`)

## License

AGPL-3.0, see `LICENSE`.
