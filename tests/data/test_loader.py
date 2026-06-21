import pytest
import torch

from snapdragon.data.batches import BatchBuilder
from snapdragon.data.loader import TrainLoader
from snapdragon.data.memmap_dataset import TokenDataset


@pytest.fixture
def ds_path(tmp_path, make_token_dataset):
    make_token_dataset(tmp_path, 6001)
    return tmp_path


def _builder(path):
    return BatchBuilder([TokenDataset(path)], [1.0], seq_len=16, micro_batch_size=4, seed=3)


def test_iterates_in_order(ds_path):
    b = _builder(ds_path)
    loader = TrainLoader(b)
    first = next(loader)
    assert torch.equal(first.x, b.get_batch(0)[0])
    assert loader.batch_index == 1


def test_resume_reproduces_the_stream(ds_path):
    loader = TrainLoader(_builder(ds_path))
    for _ in range(5):
        next(loader)
    state = loader.state_dict()
    expected = [next(loader).x for _ in range(3)]

    resumed = TrainLoader(_builder(ds_path))
    resumed.load_state_dict(state)
    for want in expected:
        assert torch.equal(next(resumed).x, want)
