import urllib.error

import pytest

from snapdragon.eval.data import fetch


def test_fetch_downloads_once_then_uses_the_cache(tmp_path):
    src = tmp_path / "remote.jsonl"
    src.write_text("first")
    cache = tmp_path / "cache"
    a = fetch(src.as_uri(), cache)
    assert a.read_text() == "first"
    src.write_text("changed")  # the cache must win: no second download
    assert fetch(src.as_uri(), cache).read_text() == "first"


def test_filename_override_and_cache_dir_creation(tmp_path):
    src = tmp_path / "x.txt"
    src.write_text("hi")
    out = fetch(src.as_uri(), tmp_path / "deep" / "dir", filename="renamed.txt")
    assert out.name == "renamed.txt"
    assert out.read_text() == "hi"


def test_failed_download_leaves_no_partial_file(tmp_path):
    with pytest.raises(urllib.error.URLError):
        fetch((tmp_path / "missing.jsonl").as_uri(), tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []
