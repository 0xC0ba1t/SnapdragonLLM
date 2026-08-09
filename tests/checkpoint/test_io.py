import pytest
import torch

from snapdragon.checkpoint.io import COMPLETE, list_checkpoints, save_checkpoint, step_dir_name


def test_save_writes_components_and_marker(tmp_path):
    path = save_checkpoint(tmp_path, 500, {"model": {"w": torch.ones(2)}, "meta": {"a": 1}})
    assert path.name == "step_00000500"
    assert (path / "model.pt").exists()
    assert (path / "meta.pt").exists()
    assert (path / COMPLETE).read_text() == "500"
    assert torch.load(path / "model.pt")["w"].tolist() == [1.0, 1.0]


def test_no_temp_dirs_left_behind(tmp_path):
    save_checkpoint(tmp_path, 1, {"x": 1})
    assert [p.name for p in tmp_path.iterdir()] == [step_dir_name(1)]


def test_list_is_sorted_and_ignores_incomplete_and_foreign(tmp_path):
    for s in (30, 10, 20):
        save_checkpoint(tmp_path, s, {"x": s})
    (tmp_path / "step_00000040").mkdir()  # looks like a checkpoint but has no COMPLETE marker
    (tmp_path / "notes").mkdir()
    assert [s for s, _ in list_checkpoints(tmp_path)] == [10, 20, 30]


def test_list_on_missing_dir(tmp_path):
    assert list_checkpoints(tmp_path / "nope") == []


def test_overwriting_a_step_replaces_it(tmp_path):
    save_checkpoint(tmp_path, 5, {"x": 1})
    path = save_checkpoint(tmp_path, 5, {"x": 2})
    assert torch.load(path / "x.pt") == 2


def test_failed_save_leaves_nothing_visible(tmp_path):
    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with pytest.raises(Exception, match="boom"):
        save_checkpoint(tmp_path, 7, {"ok": 1, "bad": Unpicklable()})
    assert list_checkpoints(tmp_path) == []
    assert list(tmp_path.iterdir()) == []
