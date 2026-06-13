import pytest

from snapdragon.tokenizer.bpe_reference import BPETokenizer

CORPUS = ["the quick brown fox jumps over the lazy dog. " * 20, "lower lowest low " * 30, "café 你好 🙂 " * 10]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=256 + 7 + 60)


@pytest.mark.parametrize(
    "text",
    ["", "hello world", "the lazy dog", "naïve café", "你好世界", "emoji 🙂🙂", "tabs\tand\nnewlines\n\n", "  spaces  "],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_merges_shorten_common_text(tok):
    text = "the quick brown fox"
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_untrained_text_falls_back_to_bytes(tok):
    ids = tok.encode("zqxj")
    assert all(i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == "zqxj"


def test_vocab_size(tok):
    assert tok.vocab_size == 256 + len(tok.merges)
