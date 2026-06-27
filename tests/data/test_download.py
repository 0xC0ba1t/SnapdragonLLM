import json

import pytest

from snapdragon.data.download import iter_documents, parse_spec, with_retries


def test_parse_specs():
    assert parse_spec("hf:HuggingFaceFW/fineweb-edu:sample-10BT") == ("hf", ["HuggingFaceFW/fineweb-edu", "sample-10BT"])
    assert parse_spec("hf:wikitext") == ("hf", ["wikitext"])
    assert parse_spec("files:a.txt,b.jsonl") == ("files", ["a.txt", "b.jsonl"])


@pytest.mark.parametrize("bad", ["", "hf", "hf:", "ftp:x", "nonsense"])
def test_bad_specs(bad):
    with pytest.raises(ValueError, match="bad dataset spec"):
        parse_spec(bad)


def test_local_files(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    assert list(iter_documents(f"files:{tmp_path / 'a.jsonl'}")) == ["hello"]


def test_retries_resume_without_duplicates(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        for i in range(6):
            if calls["n"] == 1 and i == 3:
                raise ConnectionError("dropped")
            yield f"doc{i}"

    assert list(with_retries(flaky)) == [f"doc{i}" for i in range(6)]
    assert calls["n"] == 2


def test_retries_give_up(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def always_fails():
        raise OSError("down")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="giving up"):
        list(with_retries(always_fails, attempts=2))
