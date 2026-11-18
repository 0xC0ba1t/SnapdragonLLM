import pytest
import torch

from snapdragon.sft.loader import SFTLoader
from snapdragon.sft.packing import PackedRows


def _rows(n=10, t=4):
    x = torch.arange(n * t).view(n, t)
    return PackedRows(x, x + 1, torch.zeros_like(x))


def test_batch_shapes_and_alignment():
    b = next(SFTLoader(_rows(), micro_batch_size=3))
    assert b.x.shape == b.y.shape == b.position_ids.shape == (3, 4)
    assert torch.equal(b.y, b.x + 1)  # rows travel together


def test_every_row_is_seen_once_per_epoch():
    loader = SFTLoader(_rows(10), micro_batch_size=5, seed=1)
    seen = torch.cat([next(loader).x[:, 0] for _ in range(2)]).tolist()  # 2 batches = one epoch
    assert sorted(seen) == [i * 4 for i in range(10)]
    second = torch.cat([next(loader).x[:, 0] for _ in range(2)]).tolist()
    assert sorted(second) == sorted(seen)
    assert second != seen  # reshuffled


def test_resume_reproduces_the_stream():
    a = SFTLoader(_rows(), 2, seed=3)
    for _ in range(4):
        next(a)
    state = a.state_dict()
    want = [next(a).x for _ in range(3)]
    b = SFTLoader(_rows(), 2, seed=3)
    b.load_state_dict(state)
    assert all(torch.equal(next(b).x, w) for w in want)


def test_ranks_partition_a_global_batch():
    solo = SFTLoader(_rows(), 4, seed=2)
    ranks = [SFTLoader(_rows(), 2, seed=2, rank=r, world_size=2) for r in range(2)]
    for _ in range(3):
        assert torch.equal(next(solo).x, torch.cat([next(r).x for r in ranks]))


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        SFTLoader(PackedRows(*(torch.empty(0, 4, dtype=torch.long) for _ in range(3))), 2)


def test_state_survives_a_change_of_world_size():
    pair = [SFTLoader(_rows(), 2, seed=5, rank=r, world_size=2) for r in range(2)]
    for _ in range(3):
        for ld in pair:
            next(ld)
    state = pair[0].state_dict()
    assert state == {"batch_index": 3, "global_samples": 12}

    solo = SFTLoader(_rows(), 4, seed=5)
    solo.load_state_dict(state)
    reference = SFTLoader(_rows(), 4, seed=5)
    for _ in range(3):
        next(reference)
    assert torch.equal(next(solo).x, next(reference).x)  # continues the same global stream


def test_legacy_state_without_global_samples_still_loads():
    ld = SFTLoader(_rows(), 2)
    ld.load_state_dict({"batch_index": 4})
    assert ld.batch_index == 4
