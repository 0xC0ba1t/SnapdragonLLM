# Model card: SnapdragonLLM

> **Status.** The **25M** preset has been trained and evaluated (results below, measured). The **500M**
> model is code complete and configured, but its weights have **not** been trained; every field in its
> section is empty on purpose.

Both models share one code base, tokenizer and training recipe; only the size differs.

## SnapdragonLLM-25M (trained)

A small base model, trained to check that the whole pipeline (data, tokenizer, training, resume,
evaluation, sampling) works on real hardware. It is **not useful for anything beyond that**; see
[what it can and cannot do](#what-it-can-and-cannot-do).

### Model

* Decoder-only transformer, 25.2M parameters: 8 layers, width 384, 6 query / 2 key-value heads (GQA),
  SwiGLU MLP (1024), RMSNorm + QK-norm, RoPE (theta 500k), tied embeddings, 2048 context.
* Custom 32,768-token byte-level BPE tokenizer trained from scratch on about 0.5 GB of FineWeb-Edu text.

### Training run

| | |
|---|---|
| config | `configs/train/pretrain-25m.yaml` with `train.precision=fp32 train.micro_batch_size=8` |
| data | FineWeb-Edu `sample-10BT`, first 520M tokens: 510.0M train (499,586 documents) and 10.0M validation (9,786); quality-filtered and exact-deduplicated |
| tokens seen | 0.5B (3,814 steps of 131,072 tokens, about 20 tokens per parameter) |
| batch | 131,072 tokens per step: 8 sequences x 2048 x 2 GPUs x 4 gradient-accumulation steps |
| sequence length | 512 growing to 2048 over the first 50M tokens, then 2048 |
| optimizer | Muon (hidden matrices, lr 0.02) + 8-bit AdamW (embeddings and gains, lr 1e-3) |
| schedule | warmup 200 steps, constant, then decay over the last 20% (WSD) |
| precision | fp32 (the T4 has no bf16 or flash-attention hardware), `torch.compile`, DDP |
| hardware | two Tesla T4 (Kaggle), about 11.6 GB peak per GPU, 21-29k tokens/s |
| wall time | roughly 6-7 hours of training in two sessions; the first ended after the step-1500 checkpoint and the run was resumed from it |

### Results

Measured with `scripts/eval.py --precision fp32` on one T4, final checkpoint (step 3814).

| benchmark | n | metric | result | chance |
|---|---|---|---|---|
| validation perplexity (FineWeb-Edu) | 409,600 tokens | ppl / loss | **36.2** / 3.59 | - |
| HellaSwag | 10,042 | acc / acc_norm | **26.8%** / 26.5% | 25% |
| ARC-Easy | 2,376 | acc / acc_norm | **42.4%** / 37.3% | about 25% |
| PIQA | 1,838 | acc / acc_norm | **57.8%** / 57.5% | 50% |
| LAMBADA | 5,153 | acc / ppl | **15.2%** / 77.9 | about 0% |
| ARC-Challenge | 1,172 | acc / acc_norm | **17.8%** / 22.7% | about 25% |

The model is clearly above chance only on ARC-Easy and PIQA, barely above it on HellaSwag, and at chance on
ARC-Challenge (that test was built from questions that simple word-statistics methods get wrong, so a model
with no reasoning ability lands there). These are in line with what models of this size usually reach, but
no comparison against published numbers has been made here.

### Samples

Unedited, from the final checkpoint (`seed 1`, `top-p 0.9`, `repetition penalty 1.1`, at most 80 new
tokens). The first two have a temperature of 0.9, the third 0.7. They are included because they are typical.

> **The water cycle is the process by which** water is supplied, and the hydrologic phases of the life cycle
> are identified. The more complex the hydrological cycles and the more complete the process of ad-hoc life.
> In these three case studies, the roles of hydrologic stem cells, i.e. cell replication (hacking), cellular
> respiration, and metabolic processes, were all explained in a paper entitled "Planativ. Biological

> **In 1969, the first humans** of this ship, the Chairman Flushing had been in the seas with many ships and
> traders. The Chairman was the only Europeans that made the advent of steamboatboats, but was still their
> home, on 14 March 1807. The scene of the ship as they came in the month of May 8, 1794, became one of the
> most important. Its

> **To solve a quadratic equation, you** will need to know the difference between the two. The sum of the two
> equations is 1. The sum of the two equations is 2. The formula for the sum of the two equations is 3. The
> equation of the two equations is 4. The equation of the two equations is 5. The equation of the two
> equations is 6. The equation of the two equations is

At temperature 1.1 the same prompts fall apart into word salad, and at 0.3 the water-cycle prompt repeats a
single sentence. The best range found was temperature 0.7-0.9 with a repetition penalty around 1.1-1.15.

### What it can and cannot do

* **It learned the style of educational web text:** grammatical sentences, paragraph shape, and vocabulary
  that fits the subject. It also learned where documents end (it sometimes stops on its own).
* **It has almost no knowledge and does not follow the meaning of a prompt.** "In 1969" became an
  eighteenth-century ship story, and the quadratic-equation prompt produced the right words with no
  method. Facts it states should be treated as invented.
* It is a base model: it continues text and does not answer questions or follow instructions.
* Changing sampling settings trades one failure for another (loops at low temperature, nonsense at high);
  the limit is the size of the model and the amount of data, not the sampler.

### Chat fine-tune (experiment)

The pretrained 25M weights were also fine-tuned on `yahma/alpaca-cleaned` (`configs/train/sft-25m.yaml`, about
17 minutes on the same two T4s; see [sft.md](sft.md)). Replies take the shape of an answer, but the content
is still wrong: asked for the capital of France it wrote about Napoleon without saying Paris, and a request
for a poem produced a paragraph about greenhouse gases. No benchmark was run on the fine-tuned model, and it
is not recommended for any use.

### What the first real run found

Running on real GPUs turned up problems that the CPU tests could not, all fixed in this repository:
selective checkpointing saved the wrong matmul op under the math attention fallback, and it could not be
combined with `torch.compile`; the memory estimator was optimistic for fp32 and small models; DDP and FSDP
picked their devices from CUDA availability instead of from the model; resuming from a checkpoint on a GPU
failed because the saved RNG state was moved off the CPU; micro-batch 16 ran out of memory only once the
sequence length reached 2048 (the preset now uses 8); the HellaSwag download link had been removed; and
seeded sampling on a GPU used a CPU random generator.

## SnapdragonLLM-500M (not trained)

* Decoder-only transformer, 528M parameters, 28 layers, width 1280, 20 query / 5 key-value heads (GQA),
  SwiGLU MLP (3456), RMSNorm + QK-norm, RoPE (theta 500k), tied embeddings, 2048 context.
* Same tokenizer design (32,768-token byte-level BPE; special tokens `<|bos|> <|eos|> <|pad|> <|system|>
  <|user|> <|assistant|> <|end_turn|>`).
* Details: [architecture.md](architecture.md). Hugging Face loadable as `Qwen3ForCausalLM` after
  `scripts/export_model.py --format hf`.

### Training recipe (as configured in `configs/train/pretrain-16gb.yaml`)

| | |
|---|---|
| data | FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`), quality-filtered and exact-deduplicated |
| tokens | 10B (about 19 tokens per parameter, close to the Chinchilla-optimal ~20) |
| batch | 524,288 tokens per step, context 2048, sequence-length warmup 512 to 2048 over 0.5B tokens |
| optimizer | Muon (hidden matrices, lr 0.02) + 8-bit AdamW (embeddings/gains, lr 6e-4) |
| schedule | warmup 500 steps, constant, then decay over the last 20% (WSD) to 10% |
| precision | bf16 autocast, fp32 master weights |
| hardware | designed for one 16 GB GPU; see [training-on-a-16gb-gpu.md](training-on-a-16gb-gpu.md) |

### Evaluation

Results are intentionally blank until trained:

| benchmark | metric | result |
|---|---|---|
| validation perplexity (FineWeb-Edu val) | ppl | - |
| HellaSwag | acc_norm | - |
| ARC-Easy / ARC-Challenge | acc_norm | - / - |
| PIQA | acc_norm | - |
| LAMBADA | acc / ppl | - / - |

For orientation only: models of this size trained on ~10B tokens typically score in the low-to-mid 30s on
HellaSwag `acc_norm`. Treat that as context, not as a claim about this model. The 25M run above reached
26.5%.

## Intended use

Research and education on small-scale LLM pretraining, and as a base for fine-tuning experiments on
consumer hardware. Not intended for production decision-making, medical/legal/financial advice, or any
setting where errors have serious consequences.

## Limitations and risks

* Trained only on English web text; limited multilingual ability, and no safety tuning in the base models.
* Small models hallucinate facts, make arithmetic and reasoning errors, and can reproduce biases and
  toxic content present in web data. The 25M model does this almost constantly (see the samples).
* The SFT recipe (`docs/sft.md`) teaches a chat format; it does not make outputs reliable or safe.
* Context length is 2048 tokens.

## License

Code: AGPL-3.0 (see `LICENSE`). Check the licenses of any dataset you train on; FineWeb-Edu is ODC-By.
