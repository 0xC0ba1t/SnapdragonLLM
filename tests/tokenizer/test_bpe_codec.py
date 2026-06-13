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
    assert tok.vocab_size == 256 + len(tok.merges) + 7


def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    text = "the quick brown fox"
    assert loaded.encode(text) == tok.encode(text)


def test_load_rejects_other_files(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"type": "something-else"}')
    with pytest.raises(ValueError, match="not a reference BPE"):
        BPETokenizer.load(path)


def test_special_tokens_encode_to_single_ids(tok):
    ids = tok.encode("<|bos|>hello<|eos|>")
    assert ids[0] == tok.special_ids["<|bos|>"]
    assert ids[-1] == tok.special_ids["<|eos|>"]
    assert tok.decode(ids) == "<|bos|>hello<|eos|>"


def test_special_tokens_can_be_treated_as_text(tok):
    ids = tok.encode("<|bos|>", allow_special=False)
    assert tok.special_ids["<|bos|>"] not in ids
    assert tok.decode(ids) == "<|bos|>"
