"""Both tokenizers follow the same design; check the contract they must share."""
import random

import pytest

from snapdragon.tokenizer.bpe_reference import BPETokenizer
from snapdragon.tokenizer.fast import FastTokenizer, train_fast_tokenizer
from snapdragon.tokenizer.special_tokens import special_token_ids

V = 256 + 7 + 100


@pytest.fixture(scope="module")
def corpus():
    rng = random.Random(4)
    words = ["".join(rng.choice("abcdefgh") for _ in range(rng.randint(3, 8))) for _ in range(150)]
    return [" ".join(rng.choice(words) for _ in range(10)) for _ in range(600)]


def test_special_ids_agree_with_the_spec(corpus):
    ref = BPETokenizer.train(corpus, V)
    fast = FastTokenizer(train_fast_tokenizer(corpus, V))
    expected = special_token_ids(V)
    assert ref.special_ids == expected
    assert fast.bos_id == expected["<|bos|>"]
    assert fast.eos_id == expected["<|eos|>"]
    assert ref.vocab_size == fast.vocab_size == V


def test_both_roundtrip_the_same_text(corpus):
    ref = BPETokenizer.train(corpus, V)
    fast = FastTokenizer(train_fast_tokenizer(corpus, V))
    for text in ["abc def", "café 你好", "  tabs\tand\n\nlines  "]:
        assert ref.decode(ref.encode(text)) == text
        assert fast.decode(fast.encode(text)) == text
