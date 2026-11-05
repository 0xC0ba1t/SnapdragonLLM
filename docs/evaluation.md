# Evaluation

All evaluation goes through `scripts/eval.py`, or runs inside training via callbacks.

## Benchmarks

Multiple-choice tasks are scored the way lm-evaluation-harness and the HellaSwag paper do it: each choice
is appended to the context and the model's log-likelihood of the **choice tokens only** is computed.

* `acc` - the choice with the highest total log-likelihood.
* `acc_norm` - the choice with the highest log-likelihood *per token*, so a long answer is not penalised
  for being long. This is the number usually quoted for HellaSwag.

| task | data | notes |
|---|---|---|
| `hellaswag` | rowanz/hellaswag val (10,042) | wikiHow artefacts cleaned as in the harness |
| `arc_easy`, `arc_challenge` | `allenai/ai2_arc` test | needs the `data` extra |
| `piqa` | PIQA validation | questions and labels are separate files |
| `lambada` | OpenAI LAMBADA test | accuracy needs *every* token of the last word; also perplexity |

Files are downloaded once into `~/.cache/snapdragon/eval` (override with `SNAPDRAGON_CACHE`); a
partial download never leaves a corrupt cache entry.

    python scripts/eval.py --checkpoint runs/snapdragon-500m-16gb/checkpoints \
        --tokenizer data/tokenizer/snapdragon-32k.json \
        --tasks hellaswag arc_easy arc_challenge piqa lambada --out results.json

Use `--limit 200` for a quick look. These tasks are *relative* signals during training; they are noisy on
small subsets and a 500M model trained on 10B tokens will not beat much larger models.

## Perplexity and bits per byte

    python scripts/eval.py --checkpoint runs/x/checkpoints --perplexity data/fineweb-edu

evaluates the dataset's `val` split with the same fused loss as training. `PerplexityResult.bits_per_byte`
divides by the tokenizer's bytes-per-token so models with different vocabularies can be compared.

## During training

* `train.eval.every_steps` - validation loss on the `val` split (`val_loss`, `val_ppl`, and
  `val_loss_ema` if `train.ema_decay > 0`).
* `train.eval.benchmark_every` + `train.eval.tasks` - benchmark tasks every N steps. Off by default
  because they download data; they run on rank 0 only.

## Correctness checks that back these numbers

* The scoring functions are unit-tested against hand-computed losses and a fake "oracle" model.
* The exported model matches `transformers` Qwen3 to 1e-4 in logits and token-for-token in greedy
  generation, so published numbers for the same weights in other harnesses should agree.
