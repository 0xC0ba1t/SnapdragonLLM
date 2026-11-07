# Model card: SnapdragonLLM-500M

> **Status: code complete, weights not yet trained.** This card describes the model that the code builds
> and the recipe that is configured. Every performance field below is deliberately empty until a real run
> has produced it; nothing here is a measured result.

## Model

* Decoder-only transformer, 528M parameters, 28 layers, width 1280, 20 query / 5 key-value heads (GQA),
  SwiGLU MLP (3456), RMSNorm + QK-norm, RoPE (theta 500k), tied embeddings, 2048 context.
* Custom 32,768-token byte-level BPE tokenizer trained from scratch (special tokens `<|bos|> <|eos|>
  <|pad|> <|system|> <|user|> <|assistant|> <|end_turn|>`).
* Details: [architecture.md](architecture.md). Hugging Face loadable as `Qwen3ForCausalLM` after
  `scripts/export_model.py --format hf`.

## Intended use

Research and education on small-scale LLM pretraining, and as a base for fine-tuning experiments on
consumer hardware. Not intended for production decision-making, medical/legal/financial advice, or any
setting where errors have serious consequences.

## Training recipe (as configured in `configs/train/pretrain-16gb.yaml`)

| | |
|---|---|
| data | FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`), quality-filtered and exact-deduplicated |
| tokens | 10B (about 19 tokens per parameter, close to the Chinchilla-optimal ~20) |
| batch | 524,288 tokens per step, context 2048, sequence-length warmup 512 to 2048 over 0.5B tokens |
| optimizer | Muon (hidden matrices, lr 0.02) + 8-bit AdamW (embeddings/gains, lr 6e-4) |
| schedule | warmup 500 steps, constant, then decay over the last 20% (WSD) to 10% |
| precision | bf16 autocast, fp32 master weights |
| hardware | designed for one 16 GB GPU; see [training-on-a-16gb-gpu.md](training-on-a-16gb-gpu.md) |

## Evaluation

Run `scripts/eval.py` (see [evaluation.md](evaluation.md)). Results are intentionally blank until trained:

| benchmark | metric | result |
|---|---|---|
| validation perplexity (FineWeb-Edu val) | ppl | - |
| HellaSwag | acc_norm | - |
| ARC-Easy / ARC-Challenge | acc_norm | - / - |
| PIQA | acc_norm | - |
| LAMBADA | acc / ppl | - / - |

For orientation only: models of this size trained on ~10B tokens typically score in the low-to-mid 30s on
HellaSwag `acc_norm`. Treat that as context, not as a claim about this model.

## Limitations and risks

* Trained only on English web text; limited multilingual ability, and no safety tuning in the base model.
* A 500M model hallucinates facts, makes arithmetic and reasoning errors, and can reproduce biases and
  toxic content present in web data.
* The SFT recipe (`docs/sft.md`) teaches a chat format; it does not make outputs reliable or safe.
* Context length is 2048 tokens.

## License

Code: AGPL-3.0 (see `LICENSE`). Check the licenses of any dataset you train on; FineWeb-Edu is ODC-By.
