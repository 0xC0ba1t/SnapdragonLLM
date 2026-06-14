import random

import pytest

from snapdragon.tokenizer.fast import FastTokenizer, train_fast_tokenizer
from snapdragon.tokenizer.special_tokens import SPECIAL_TOKENS


def _corpus():
    rng = random.Random(0)
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 7))) for _ in range(300)]
    lines = [" ".join(rng.choice(words) for _ in range(12)) for _ in range(400)]
    return [*lines, "the quick brown fox. café 你好 🙂 " * 20]


CORPUS = _corpus()
VOCAB = 256 + 7 + 80


@pytest.fixture(scope="module")
def tok():
    return FastTokenizer(train_fast_tokenizer(CORPUS, VOCAB))


def test_vocab_size_and_special_ids(tok):
    assert tok.vocab_size == VOCAB
    first = VOCAB - len(SPECIAL_TOKENS)
    assert tok.bos_id == first
    assert tok.pad_id == first + 2


@pytest.mark.parametrize("text", ["", "hello world", "the lazy dog", "naïve café", "你好世界", "emoji 🙂", "a\n\n b\t c"])
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_special_token_is_one_id(tok):
    assert tok.encode("<|eos|>") == [tok.eos_id]


def test_batch_matches_single(tok):
    texts = ["hello world", "the quick fox"]
    assert tok.encode_batch(texts) == [tok.encode(t) for t in texts]


def test_save_and_load(tok, tmp_path):
    path = tmp_path / "sub" / "tok.json"
    tok.save(path)
    again = FastTokenizer.from_file(path)
    assert again.encode("the quick fox") == tok.encode("the quick fox")


def test_iter_encode_streams_in_batches(tok):
    out = list(tok.iter_encode((f"line {i}" for i in range(5)), batch_size=2))
    assert len(out) == 5


def test_corpus_too_small_is_an_error():
    with pytest.raises(ValueError, match="corpus too small"):
        train_fast_tokenizer(["ab"], VOCAB)
