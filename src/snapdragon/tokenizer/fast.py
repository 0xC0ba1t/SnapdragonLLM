"""Fast byte-level BPE built on HuggingFace ``tokenizers`` (Rust). Used for real data prep."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from snapdragon.tokenizer.pretokenize import PATTERN
from snapdragon.tokenizer.special_tokens import BOS, EOS, PAD, SPECIAL_TOKENS


def _build_empty() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PATTERN.pattern), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def train_fast_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    """Train a byte-level BPE with ``vocab_size`` ids in total; specials take the last ids."""
    tok = _build_empty()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    got = tok.get_vocab_size()
    if got != vocab_size - len(SPECIAL_TOKENS):
        raise ValueError(f"corpus too small: learned {got} tokens, wanted {vocab_size - len(SPECIAL_TOKENS)}")
    tok.add_special_tokens(list(SPECIAL_TOKENS))
    return tok


class FastTokenizer:
    """Thin wrapper exposing the small API the rest of the codebase needs."""

    def __init__(self, tok: Tokenizer) -> None:
        self.tok = tok
        self.bos_id = tok.token_to_id(BOS)
        self.eos_id = tok.token_to_id(EOS)
        self.pad_id = tok.token_to_id(PAD)
        if None in (self.bos_id, self.eos_id, self.pad_id):
            raise ValueError("tokenizer is missing required special tokens")

    @classmethod
    def from_file(cls, path: str | Path) -> FastTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tok.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special)

    def iter_encode(self, texts: Iterable[str], batch_size: int = 256) -> Iterator[list[int]]:
        batch: list[str] = []
        for t in texts:
            batch.append(t)
            if len(batch) == batch_size:
                yield from self.encode_batch(batch)
                batch = []
        if batch:
            yield from self.encode_batch(batch)
