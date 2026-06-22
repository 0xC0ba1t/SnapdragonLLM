import pytest
import torch

from snapdragon.data.batches import BatchBuilder
from snapdragon.data.loader import TrainLoader
from snapdragon.data.memmap_dataset import TokenDataset
from snapdragon.data.prefetch import PrefetchLoader


@pytest.fixture
def ds_path(tmp_path, make_token_dataset):
    make_token_dataset(tmp_path, 6001)
    return tmp_path


def _builder(path):
    return BatchBuilder([TokenDataset(path)], [1.0], seq_len=16, micro_batch_size=4, seed=3)


def test_matches_synchronous_loader(ds_path):
    sync = TrainLoader(_builder(ds_path))
    with PrefetchLoader(_builder(ds_path), depth=3) as pre:
        for _ in range(10):
            assert torch.equal(next(pre).x, next(sync).x)


def test_batch_index_counts_consumed_not_prefetched(ds_path):
    with PrefetchLoader(_builder(ds_path), depth=4) as pre:
        for _ in range(3):
            next(pre)
        assert pre.state_dict() == {"batch_index": 3}


def test_start_batch(ds_path):
    expected = _builder(ds_path).get_batch(7)[0]  # a separate builder: Sampler is not thread safe
    with PrefetchLoader(_builder(ds_path), start_batch=7, depth=2) as pre:
        assert torch.equal(next(pre).x, expected)


def test_worker_errors_propagate(ds_path):
    b = _builder(ds_path)
    b.datasets[0].num_tokens = 0  # make reads fail

    with PrefetchLoader(b, depth=2) as pre, pytest.raises(IndexError):
        next(pre)


def test_close_stops_the_thread(ds_path):
    pre = PrefetchLoader(_builder(ds_path), depth=1)
    next(pre)
    pre.close()
    assert not pre._thread.is_alive()


def test_pin_memory_flag_is_safe_without_cuda(ds_path):
    with PrefetchLoader(_builder(ds_path), depth=2, pin_memory=True) as pre:
        assert next(pre).x.shape == (4, 16)
