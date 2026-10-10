"""Multi-turn chat on top of the streaming generator, using the tokenizer's chat template."""
from __future__ import annotations

from collections.abc import Iterator

from snapdragon.inference.generate import stream_generate
from snapdragon.inference.sampling import SamplingConfig
from snapdragon.inference.text import Detokenizer, StopStringFilter
from snapdragon.tokenizer.chat import encode_chat
from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.tokenizer.special_tokens import END_TURN


class ChatSession:
    """Keeps the conversation and generates the assistant's next turn.

    When the history plus the reply would overflow the context window, the oldest user/assistant
    turns are dropped (the system message, if any, is always kept).
    """

    def __init__(self, model, tok: FastTokenizer, system: str | None = None,
                 sampling: SamplingConfig | None = None, max_new_tokens: int = 256) -> None:
        self.model, self.tok = model, tok
        self.sampling = sampling or SamplingConfig()
        self.max_new_tokens = max_new_tokens
        self.messages: list[dict[str, str]] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def _prompt_ids(self) -> list[int]:
        budget = self.model.cfg.max_seq_len - self.max_new_tokens
        msgs = list(self.messages)
        while True:
            ids, _ = encode_chat(self.tok, msgs, add_generation_prompt=True)
            if len(ids) <= budget:
                return ids
            droppable = [i for i, m in enumerate(msgs) if m["role"] != "system"]
            if len(droppable) <= 1:  # only the newest message is left: truncate it from the left
                return ids[:1] + ids[-(budget - 1) :]
            del msgs[droppable[0]]

    def reply_stream(self, user_text: str, seed: int | None = None) -> Iterator[str]:
        """Add a user message, yield the assistant reply chunk by chunk, then record it."""
        self.messages.append({"role": "user", "content": user_text})
        end_id = self.tok.encode(END_TURN)[0]
        detok = Detokenizer(self.tok)
        stops = StopStringFilter([END_TURN])
        parts: list[str] = []
        for token in stream_generate(self.model, self._prompt_ids(), self.max_new_tokens, self.sampling,
                                     [end_id, self.tok.eos_id], seed):
            if token in (end_id, self.tok.eos_id):
                break
            chunk = stops.feed(detok.push(token))
            if chunk:
                parts.append(chunk)
                yield chunk
        tail = stops.feed(detok.flush()) + stops.flush()
        if tail:
            parts.append(tail)
            yield tail
        self.messages.append({"role": "assistant", "content": "".join(parts)})

    def reply(self, user_text: str, seed: int | None = None) -> str:
        return "".join(self.reply_stream(user_text, seed))

    def reset(self) -> None:
        self.messages = [m for m in self.messages if m["role"] == "system"]
