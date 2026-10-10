"""Token ids <-> streaming text: incremental detokenization and stop strings."""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from snapdragon.inference.generate import stream_generate
from snapdragon.inference.sampling import SamplingConfig
from snapdragon.tokenizer.fast import FastTokenizer

REPLACEMENT = "�"


class Detokenizer:
    """Turn a stream of token ids into text without emitting half a UTF-8 character.

    Byte-level BPE can split one character across tokens (an emoji is often 3-4 tokens). Decoding such a
    prefix yields U+FFFD, so we hold output back until the decoded text no longer ends in it. Only a
    short window of recent tokens is re-decoded each step.
    """

    def __init__(self, tok: FastTokenizer, window: int = 6) -> None:
        self.tok, self.window = tok, window
        self.ids: list[int] = []
        self._emitted_ids = 0  # ids [0, _emitted_ids) are fully emitted
        self._prefix_text = ""  # decode of ids[_prefix_start:_emitted_ids]
        self._prefix_start = 0

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids[self._prefix_start :])
        if text.endswith(REPLACEMENT):
            return ""  # an incomplete character: wait for more tokens
        delta = text[len(self._prefix_text) :]
        self._emitted_ids = len(self.ids)
        self._prefix_start = max(self._emitted_ids - self.window, 0)
        self._prefix_text = self.tok.decode(self.ids[self._prefix_start : self._emitted_ids])
        return delta

    def flush(self) -> str:
        """Emit whatever is still held back (an incomplete character becomes U+FFFD)."""
        text = self.tok.decode(self.ids[self._prefix_start :])
        delta = text[len(self._prefix_text) :]
        self._prefix_text = text
        self._emitted_ids = len(self.ids)
        return delta


class StopStringFilter:
    """Cut the stream at the first occurrence of any stop string, never emitting part of one.

    Text that could still turn into a stop string (a suffix equal to a prefix of one) is held back.
    """

    def __init__(self, stops: Iterable[str]) -> None:
        self.stops = [s for s in stops if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self._pending = ""
        self.done = False

    def feed(self, text: str) -> str:
        if self.done:
            return ""
        self._pending += text
        cut = min((i for s in self.stops if (i := self._pending.find(s)) != -1), default=-1)
        if cut != -1:
            out, self._pending, self.done = self._pending[:cut], "", True
            return out
        safe = len(self._pending) - self.hold
        if safe <= 0:
            return ""
        out, self._pending = self._pending[:safe], self._pending[safe:]
        return out

    def flush(self) -> str:
        out, self._pending = ("" if self.done else self._pending), ""
        return out


def stream_text(
    model,
    tok: FastTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    sampling: SamplingConfig | None = None,
    stop_strings: Iterable[str] = (),
    seed: int | None = None,
    add_bos: bool = True,
) -> Iterator[str]:
    """Yield generated text chunks. Stops at EOS, a stop string, or the token limit."""
    ids = ([tok.bos_id] if add_bos else []) + tok.encode(prompt)
    detok, stop_filter = Detokenizer(tok), StopStringFilter(stop_strings)
    for token in stream_generate(model, ids, max_new_tokens, sampling, [tok.eos_id], seed):
        if token == tok.eos_id:
            break
        chunk = stop_filter.feed(detok.push(token))
        if chunk:
            yield chunk
        if stop_filter.done:
            return
    tail = stop_filter.feed(detok.flush())
    tail += stop_filter.flush()
    if tail:
        yield tail
