import numpy as np

from snapdragon.data.manifest import DatasetManifest
from snapdragon.data.shards import read_tokens
from snapdragon.data.tokenize_shards import tokenize_to_shards

DOCS = [f"abc def {i} ghi" * 3 for i in range(60)]


def test_val_first_then_train(small_tokenizer, tmp_path):
    m = tokenize_to_shards(DOCS, small_tokenizer, tmp_path, shard_tokens=200, val_tokens=100, batch_size=7)
    assert m.shards[0].split == "val"
    assert m.total_tokens("val") >= 100
    assert m.total_tokens("train") > 0
    m.verify(tmp_path)
    assert DatasetManifest.load(tmp_path) == m


def test_every_doc_ends_in_eos_and_tokens_roundtrip(small_tokenizer, tmp_path):
    m = tokenize_to_shards(DOCS[:5], small_tokenizer, tmp_path, shard_tokens=10_000, val_tokens=0)
    toks = np.concatenate([read_tokens(tmp_path / s.name) for s in m.shards])
    docs = np.split(toks, np.flatnonzero(toks == small_tokenizer.eos_id) + 1)[:-1]
    assert len(docs) == 5
    for d, text in zip(docs, DOCS[:5]):
        assert small_tokenizer.decode(d[:-1].tolist()) == text


def test_max_tokens_stops_early(small_tokenizer, tmp_path):
    m = tokenize_to_shards(DOCS, small_tokenizer, tmp_path, shard_tokens=10_000, val_tokens=0, max_tokens=150)
    total = m.total_tokens()
    assert 150 <= total < 150 + 100  # stopped within one document of the budget
