# Supervised fine-tuning (chat)

SFT turns the pretrained base model into one that follows a conversation format. It reuses the
pretraining `Trainer` unchanged: only the data loader and a few config values differ.

## Pipeline

1. **Normalise your data** to `{"messages": [{"role": ..., "content": ...}, ...]}` jsonl:

       python scripts/prepare_sft.py --hf yahma/alpaca-cleaned --format alpaca --out data/sft/chat.jsonl --shuffle-seed 1

   Supported inputs: `alpaca` (instruction/input/output), `sharegpt` (conversations/from/value) and
   `messages` (OpenAI style). Conversations with an empty turn, that do not start with the user and end
   with the assistant, or that do not alternate, are dropped.

2. **Train** from the pretrained weights:

       python scripts/train.py --config configs/train/sft-500m-16gb.yaml

   `train.init_from` loads *weights only* (the architecture must match; mismatching fields are reported)
   and starts a fresh optimizer, schedule and data position. Re-running the command resumes this SFT run.

3. **Chat**: `python scripts/chat.py --checkpoint runs/snapdragon-500m-sft/checkpoints --tokenizer ...`

## Format and masking

The tokenizer's chat template is `<|bos|>` then, per turn, `<|system|>|<|user|>|<|assistant|>`, the
content, and `<|end_turn|>`. The loss is applied **only to assistant content and its `<|end_turn|>`**
(the end-of-turn token is trained so the model learns to stop). System and user tokens, padding, and the
first token of every packed example are masked with `-100`.

## Packing

Examples are packed greedily into rows of `seq_len + 1` tokens so no compute goes to padding. Examples are
never split across rows, and a conversation longer than a row is skipped rather than truncated (cutting
an answer in half would teach the model to stop mid-sentence). Position ids restart at every example.

## Recipe notes

* Learning rates are lower than pretraining (`muon_lr 0.004`, `adam_lr 1e-4`) with a short warmup and a
  cosine decay: fine-tuning should nudge the model, not overwrite it.
* The global batch is 65,536 tokens (8 micro-batches of 4 x 2048 on a 16 GB card).
* `max_tokens` counts *rows seen*, so a small dataset is repeated for several epochs; watch for
  overfitting beyond ~3 epochs.
* Data quality dominates. 10k clean conversations beat 500k noisy ones at this scale.
