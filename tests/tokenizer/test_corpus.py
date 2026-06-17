import json

from snapdragon.tokenizer.corpus import iter_local_texts, take_bytes


def test_jsonl_and_txt(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"text": "one"}) + "\n\n" + json.dumps({"text": "two"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text("three", encoding="utf-8")
    out = list(iter_local_texts([tmp_path / "a.jsonl", tmp_path / "b.txt"]))
    assert out == ["one", "two", "three"]


def test_custom_field(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"body": "x"}) + "\n", encoding="utf-8")
    assert list(iter_local_texts([tmp_path / "a.jsonl"], field="body")) == ["x"]


def test_take_bytes_keeps_crossing_document():
    docs = ["aaaa", "bbbb", "cccc", "dddd"]
    assert list(take_bytes(docs, 5)) == ["aaaa", "bbbb"]


def test_take_bytes_counts_utf8_bytes():
    assert list(take_bytes(["é" * 3, "x"], 6)) == ["ééé"]
