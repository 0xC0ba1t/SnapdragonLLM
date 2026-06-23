import pytest

from snapdragon.config.data import SourceConfig
from snapdragon.config.schema import Config
from snapdragon.data.factory import build_train_loader, build_val_loader
from snapdragon.data.manifest import DatasetManifest, ShardInfo
from snapdragon.data.shards import ShardWriter


@pytest.fixture
def cfg(tmp_path):
    with ShardWriter(tmp_path, "train", 10**6, 65000) as w:
        w.add_document(list(range(5000)))
    with ShardWriter(tmp_path, "val", 10**6, 65000) as v:
        v.add_document(list(range(2000)))
    m = DatasetManifest(vocab_size=65001, eos_id=65000)
    m.shards = [ShardInfo(s, t, d, "train") for s, t, d in w.files] + [
        ShardInfo(s, t, d, "val") for s, t, d in v.files
    ]
    m.save(tmp_path)
    c = Config()
    c.model.vocab_size = 65001
    c.model.max_seq_len = c.train.seq_len = c.data.seq_len = 64
    c.train.micro_batch_size = 2
    c.data.sources = [SourceConfig("main", str(tmp_path))]
    c.data.pin_memory = False
    return c


def test_train_loader_shapes(cfg):
    with build_train_loader(cfg) as loader:
        b = next(loader)
        assert b.x.shape == (2, 64)


def test_val_loader_uses_val_split(cfg):
    v = build_val_loader(cfg)
    assert len(v) > 0
    assert next(iter(v)).x.shape == (2, 64)


def test_empty_sources_is_a_clear_error():
    with pytest.raises(ValueError, match="data.sources is empty"):
        build_train_loader(Config())
