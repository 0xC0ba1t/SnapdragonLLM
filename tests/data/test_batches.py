import pytest
import torch

from snapdragon.data.batches import BatchBuilder
from snapdragon.data.manifest import DatasetManifest, ShardInfo
from snapdragon.data.memmap_dataset import TokenDataset
from snapdragon.data.shards import ShardWriter


def make_dataset(path, n_tokens, offset=0):
    with ShardWriter(path, "train", shard_tokens=10**6, eos_id=65000) as w:
        w.add_document(list(range(offset, offset + n_tokens - 1)))
    m = DatasetManifest(vocab_size=65001, eos_id=65000)
    m.shards = [ShardInfo(s, t, d, "train") for s, t, d in w.files]
    m.save(path)
    return TokenDataset(path)


@pytest.fixture
def ds(tmp_path):
    return make_dataset(tmp_path, 4001)


def test_shapes_and_shift(ds):
    b = BatchBuilder([ds], [1.0], seq_len=16, micro_batch_size=4)
    x, y = b.get_batch(0)
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])  # targets are inputs shifted by one


def test_batches_are_deterministic_functions_of_n(ds):
    b = BatchBuilder([ds], [1.0], seq_len=16, micro_batch_size=4, seed=5)
    first = [b.get_batch(n) for n in range(6)]
    again = BatchBuilder([ds], [1.0], seq_len=16, micro_batch_size=4, seed=5)
    for n in (5, 0, 3, 1):
        assert torch.equal(again.get_batch(n)[0], first[n][0])


def test_ranks_together_equal_single_process(ds):
    single = BatchBuilder([ds], [1.0], seq_len=16, micro_batch_size=8, seed=1)
    ranks = [BatchBuilder([ds], [1.0], 16, 2, seed=1, rank=r, world_size=4) for r in range(4)]
    for n in range(3):
        want = single.get_batch(n)[0]
        got = torch.cat([r.get_batch(n)[0] for r in ranks])
        assert torch.equal(want, got)


def test_two_sources_are_both_used(tmp_path):
    a = make_dataset(tmp_path / "a", 2001, offset=0)
    b = make_dataset(tmp_path / "b", 2001, offset=30000)
    bb = BatchBuilder([a, b], [1, 1], seq_len=16, micro_batch_size=16, seed=0)
    x, _ = bb.get_batch(0)
    firsts = x[:, 0].numpy()
    assert (firsts < 30000).any()
    assert (firsts >= 30000).any()
    assert torch.all(x[:, 1:] - x[:, :-1] == 1)  # windows are contiguous token runs
