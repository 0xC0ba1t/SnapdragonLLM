import random

import pytest

from snapdragon.tokenizer.fast import FastTokenizer, train_fast_tokenizer
from snapdragon.tokenizer.metrics import Fertility, measure_fertility


def test_fertility_properties():
    f = Fertility(tokens=150, words=100, bytes=600)
    assert f.tokens_per_word == pytest.approx(1.5)
    assert f.bytes_per_token == pytest.approx(4.0)


def test_empty_input_does_not_divide_by_zero():
    f = Fertility(tokens=0, words=0, bytes=0)
    assert f.tokens_per_word == 0.0
    assert f.bytes_per_token == 0.0


def test_trained_tokenizer_compresses_its_own_corpus():
    rng = random.Random(3)
    words = ["".join(rng.choice("abcdefgh") for _ in range(rng.randint(3, 8))) for _ in range(100)]
    corpus = [" ".join(rng.choice(words) for _ in range(10)) for _ in range(500)]
    tok = FastTokenizer(train_fast_tokenizer(corpus, 256 + 7 + 150))
    f = measure_fertility(tok, corpus[:50])
    assert f.bytes_per_token > 1.5
