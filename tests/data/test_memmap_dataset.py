import pickle

import numpy as np
import pytest

from snapdragon.data.manifest import DatasetManifest, ShardInfo
from snapdragon.data.memmap_dataset import TokenDataset
from snapdragon.data.shards import ShardWriter


@pytest.fixture
def ds_dir(tmp_path):
    """Three shards holding tokens 0..N-1 in order (eos id is out of the way)."""
    stream = list(range(100))
    with ShardWriter(tmp_path, "train", shard_tokens=34, eos_id=65000) as w:
        for i in range(0, 99, 33):
            w.add_document(stream[i : i + 33])
    m = DatasetManifest(vocab_size=65001, eos_id=65000)
    m.shards = [ShardInfo(s, t, d, "train") for s, t, d in w.files]
    m.save(tmp_path)
    return tmp_path


def test_total_size(ds_dir):
    ds = TokenDataset(ds_dir)
    assert ds.num_tokens == 3 * 34
    assert len(ds.shards) == 3


def test_read_across_shard_boundary(ds_dir):
    ds = TokenDataset(ds_dir)
    full = np.concatenate([[*range(i, i + 33), 65000] for i in range(0, 99, 33)])
    assert ds.read(30, 10).tolist() == full[30:40].tolist()
    assert ds.read(0, ds.num_tokens).tolist() == full.tolist()


def test_out_of_range(ds_dir):
    ds = TokenDataset(ds_dir)
    with pytest.raises(IndexError):
        ds.read(100, 10)


def test_windows(ds_dir):
    ds = TokenDataset(ds_dir)
    assert ds.num_windows(10) == (102 - 1) // 10
    w = ds.get_window(2, 10)
    assert w.shape == (11,)
    assert w.tolist() == ds.read(20, 11).tolist()


def test_missing_split(ds_dir):
    with pytest.raises(ValueError, match="no shards"):
        TokenDataset(ds_dir, "val")


def test_pickles_without_open_maps(ds_dir):
    ds = TokenDataset(ds_dir)
    ds.read(0, 5)
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.read(0, 5).tolist() == ds.read(0, 5).tolist()
