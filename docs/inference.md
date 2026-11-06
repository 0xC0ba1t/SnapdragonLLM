# Inference, chat and export

## Generation

    python scripts/generate.py --checkpoint runs/x/checkpoints --tokenizer data/tokenizer/snapdragon-32k.json \
        --prompt "The three laws of thermodynamics are" --max-new-tokens 120 \
        --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 --seed 1

Generation is one **prefill** pass over the prompt that fills a preallocated KV cache, then one
single-token step per new token. The cache is `(layers, batch, kv_heads, max_len, head_dim)`: for the 500M
model at 2048 context in bf16 that is 28 x 5 x 2048 x 64 x 2 (K and V) x 2 bytes = **73 MB** per sequence,
thanks to grouped-query attention.

Sampling pipeline (all optional, applied in this order): repetition penalty, temperature, top-k, top-p,
min-p. `--temperature 0` is greedy. A given `--seed` reproduces the same text.

Correctness: cached decoding is tested to produce the same logits as a full forward pass at every
position (including chunked prefill, which needs an offset causal mask), and greedy output equals
`transformers`' `generate` for the exported model.

Text streaming is UTF-8 safe: byte-level BPE can split one character across several tokens, so the
detokenizer holds output back until a character is complete. `--stop "text"` stops at a string and never
prints part of it.

## Chat

    python scripts/chat.py --checkpoint runs/snapdragon-500m-sft/checkpoints \
        --tokenizer data/tokenizer/snapdragon-32k.json --system "You are a concise assistant."

`/reset` clears the conversation, `/quit` exits. When history plus the reply would overflow the 2048-token
context, the **oldest turns are dropped** (the system message is kept). Chat only makes sense after SFT;
a base model just continues text.

## Export for release

    python scripts/export_model.py --checkpoint runs/x/checkpoints --out release/snapdragon-500m \
        --tokenizer data/tokenizer/snapdragon-32k.json --format safetensors hf

* `snapdragon/` - `model.safetensors` (bf16, tied weights stored once) + `config.json` + `tokenizer.json`;
  load with `snapdragon.export.safetensors_io.load_pretrained`.
* `hf/` - a Qwen3-format directory that `AutoModelForCausalLM.from_pretrained` loads directly. (Our
  block *is* a Qwen3 block: pre-norm GQA with per-head QK-norm, RoPE, SwiGLU, no biases.) The tokenizer
  JSON is a standard `tokenizers` file; wrapping it in a `PreTrainedTokenizerFast` needs the special
  token names `<|bos|>`, `<|eos|>`, `<|pad|>`.

safetensors cannot execute code on load (unlike pickled `.pt` checkpoints), so release artifacts use it.
