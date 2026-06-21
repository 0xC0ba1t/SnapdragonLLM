import pytest
import torch

from snapdragon.data.batches import BatchBuilder


@pytest.fixture
def ds(tmp_path, make_token_dataset):
    return make_token_dataset(tmp_path, 4001)


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


def test_two_sources_are_both_used(tmp_path, make_token_dataset):
    a = make_token_dataset(tmp_path / "a", 2001, offset=0)
    b = make_token_dataset(tmp_path / "b", 2001, offset=30000)
    bb = BatchBuilder([a, b], [1, 1], seq_len=16, micro_batch_size=16, seed=0)
    x, _ = bb.get_batch(0)
    firsts = x[:, 0].numpy()
    assert (firsts < 30000).any()
    assert (firsts >= 30000).any()
    assert torch.all(x[:, 1:] - x[:, :-1] == 1)  # windows are contiguous token runs
