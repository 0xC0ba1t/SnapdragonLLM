# Tokenizer

SnapdragonLLM uses a **32,768-token byte-level BPE** trained from scratch.

## Design

- **Byte-level**: the 256 byte values are the base alphabet, so any string round-trips and there is
  no unknown token.
- **Pre-tokenisation** (`tokenizer/pretokenize.py`): a regex splits text into words that merges never
  cross. Digits are grouped in threes so numbers don't turn into rare monolithic tokens, and both
  straight and curly apostrophes split contractions.
- **Specials last**: `<|bos|> <|eos|> <|pad|> <|system|> <|user|> <|assistant|> <|end_turn|>` take the final
  seven ids (32761-32767). Growing or shrinking the number of merges never moves them.
- 32k is a deliberate size for a 500M model: the tied embedding is ~42M parameters, about 8% of the
  model. A 128k vocabulary would triple that for little benefit at this scale.

## Two implementations

| module | use |
|---|---|
| `bpe_reference.py` | ~150 readable lines: trainer (incremental pair counts), encoder, decoder. Used as a test oracle and for learning. |
| `fast.py` | HuggingFace `tokenizers` (Rust) with the same pre-tokenizer. Used for real data preparation. |

Both are tested against the same contract (special ids, lossless round trip).

## Training it

    python scripts/train_tokenizer.py --hf HuggingFaceFW/fineweb-edu:sample-10BT --max-gb 2

Training stops reading after about `--max-gb` of text. If the corpus is too small to learn the
requested number of merges the trainer raises instead of silently producing a smaller vocabulary.

## Quality

`tokenizer.metrics.measure_fertility` reports tokens per word and bytes per token on a text sample.

## Chat format

`tokenizer/chat.py` renders `<|bos|><|role|>content<|end_turn|>...` and returns a loss mask that is 1
only on assistant content and its end-of-turn token (used by SFT).
