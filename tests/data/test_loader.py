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


def test_batch_to_device_keeps_values():
    from snapdragon.data.loader import Batch

    b = Batch(torch.arange(6).view(2, 3), torch.arange(6).view(2, 3) + 1)
    moved = b.to("cpu", non_blocking=False)
    assert torch.equal(moved.x, b.x)
    assert torch.equal(moved.y, b.y)


def test_with_doc_info_adds_position_ids():
    from snapdragon.data.loader import Batch

    x = torch.tensor([[1, 2, 9, 3, 4]])
    b = Batch(x, x.clone()).with_doc_info(eos_id=9)
    assert b.position_ids.tolist() == [[0, 1, 2, 0, 1]]
    assert b.to("cpu", non_blocking=False).position_ids.tolist() == [[0, 1, 2, 0, 1]]
