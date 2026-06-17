import random

import pytest

from snapdragon.tokenizer.chat import encode_chat, render_chat
from snapdragon.tokenizer.fast import FastTokenizer, train_fast_tokenizer

MSGS = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hi there"},
    {"role": "assistant", "content": "hello"},
]


@pytest.fixture(scope="module")
def tok():
    rng = random.Random(1)
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 7))) for _ in range(300)]
    corpus = [" ".join(rng.choice(words) for _ in range(12)) for _ in range(400)]
    return FastTokenizer(train_fast_tokenizer(corpus, 256 + 7 + 60))


def test_render_text():
    text = render_chat(MSGS)
    assert text == "<|bos|><|system|>be brief<|end_turn|><|user|>hi there<|end_turn|><|assistant|>hello<|end_turn|>"


def test_generation_prompt_appended():
    assert render_chat(MSGS[:2], add_generation_prompt=True).endswith("<|end_turn|><|assistant|>")


def test_encode_matches_rendered_text(tok):
    ids, mask = encode_chat(tok, MSGS)
    assert len(ids) == len(mask)
    assert tok.decode(ids) == render_chat(MSGS)


def test_only_assistant_turn_is_trained_on(tok):
    ids, mask = encode_chat(tok, MSGS)
    trained = [i for i, m in zip(ids, mask) if m]
    assert tok.decode(trained) == "hello<|end_turn|>"


def test_bad_role_rejected():
    with pytest.raises(ValueError, match="unknown role"):
        render_chat([{"role": "robot", "content": "x"}])
