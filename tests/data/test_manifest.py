import pytest

from snapdragon.data.manifest import DatasetManifest, ShardInfo
from snapdragon.data.shards import ShardWriter


def _make(tmp_path):
    with ShardWriter(tmp_path, "train", shard_tokens=5, eos_id=9) as w:
        for _ in range(4):
            w.add_document([1, 2, 3])
    m = DatasetManifest(vocab_size=32768, eos_id=9)
    m.shards = [ShardInfo(stem, toks, docs, "train") for stem, toks, docs in w.files]
    return m


def test_roundtrip_and_totals(tmp_path):
    m = _make(tmp_path)
    m.save(tmp_path)
    again = DatasetManifest.load(tmp_path)
    assert again == m
    assert again.total_tokens("train") == 16
    assert again.total_tokens("val") == 0
    assert len(again.shards_for("train")) == 2


def test_verify_passes_then_detects_truncation(tmp_path):
    m = _make(tmp_path)
    m.verify(tmp_path)
    path = tmp_path / "train_00000.bin"
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="size"):
        m.verify(tmp_path)


def test_verify_detects_missing_file(tmp_path):
    m = _make(tmp_path)
    (tmp_path / "train_00001.idx").unlink()
    with pytest.raises(FileNotFoundError):
        m.verify(tmp_path)


def test_unknown_version_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text('{"version": 99}')
    with pytest.raises(ValueError, match="version"):
        DatasetManifest.load(tmp_path)
