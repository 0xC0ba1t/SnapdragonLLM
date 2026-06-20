import numpy as np
import pytest

from snapdragon.data.shards import ShardWriter, read_index, read_tokens

EOS = 99


def test_documents_get_eos_and_offsets(tmp_path):
    with ShardWriter(tmp_path, "t", shard_tokens=1000, eos_id=EOS) as w:
        w.add_document([1, 2, 3])
        w.add_document([4, 5])
    toks = read_tokens(tmp_path / "t_00000")
    assert toks.tolist() == [1, 2, 3, EOS, 4, 5, EOS]
    assert read_index(tmp_path / "t_00000").tolist() == [0, 4, 7]


def test_rolls_over_to_new_shard(tmp_path):
    with ShardWriter(tmp_path, "t", shard_tokens=5, eos_id=EOS) as w:
        for _ in range(4):
            w.add_document([1, 2, 3])  # 4 tokens each with eos
    assert [f[1] for f in w.files] == [8, 8]  # shards flush after crossing the threshold
    assert sum(f[2] for f in w.files) == 4


def test_dtype_is_uint16_on_disk(tmp_path):
    with ShardWriter(tmp_path, "t", shard_tokens=100, eos_id=EOS) as w:
        w.add_document([1, 2])
    assert (tmp_path / "t_00000.bin").stat().st_size == 3 * 2


def test_rejects_out_of_range_ids(tmp_path):
    w = ShardWriter(tmp_path, "t", shard_tokens=100, eos_id=EOS)
    with pytest.raises(ValueError, match="uint16"):
        w.add_document([70000])


def test_empty_close_writes_nothing(tmp_path):
    with ShardWriter(tmp_path, "t", shard_tokens=100, eos_id=EOS) as w:
        pass
    assert w.files == []
    assert np.array(list(tmp_path.iterdir())).size == 0
