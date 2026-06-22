import pytest
import torch

from snapdragon.data.val import ValLoader


@pytest.fixture
def ds(tmp_path, make_token_dataset):
    return make_token_dataset(tmp_path, 1001)


def test_covers_windows_in_order(ds):
    v = ValLoader(ds, seq_len=10, micro_batch_size=2)
    first = next(iter(v))
    assert first.x[0, 0].item() == 0
    assert first.x[1, 0].item() == 10  # second window starts right after the first
    assert len(v) == 100 // 2


def test_iteration_is_repeatable(ds):
    v = ValLoader(ds, 10, 2, max_batches=5)
    a = [b.x.clone() for b in v]
    b = [b.x.clone() for b in v]
    assert len(a) == 5
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_ranks_split_the_work(ds):
    r0 = ValLoader(ds, 10, 2, rank=0, world_size=2)
    r1 = ValLoader(ds, 10, 2, rank=1, world_size=2)
    x0, x1 = next(iter(r0)).x, next(iter(r1)).x
    assert x0[0, 0].item() == 0
    assert x1[0, 0].item() == 20  # rank 1 starts after rank 0's two windows


def test_too_small_split(tmp_path, make_token_dataset):
    small = make_token_dataset(tmp_path, 20)
    with pytest.raises(ValueError, match="too small"):
        ValLoader(small, 10, 4)
