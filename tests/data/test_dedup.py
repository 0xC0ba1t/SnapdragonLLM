from snapdragon.data.dedup import ExactDedup, doc_hash, normalize


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Hello\n\tWORLD  ") == "hello world"


def test_hash_ignores_formatting():
    assert doc_hash("Hello   world") == doc_hash("hello\nworld ")
    assert doc_hash("hello world") != doc_hash("hello there")


def test_duplicates_are_dropped_once_seen():
    d = ExactDedup()
    assert not d.is_duplicate("a b c")
    assert d.is_duplicate("A  B C")
    assert not d.is_duplicate("x y z")
    assert (d.checked, d.dropped, len(d)) == (3, 1, 2)


def test_save_and_load(tmp_path):
    d = ExactDedup()
    d.is_duplicate("first doc")
    d.is_duplicate("second doc")
    d.save(tmp_path / "seen.bin")
    again = ExactDedup.load(tmp_path / "seen.bin")
    assert len(again) == 2
    assert again.is_duplicate("FIRST doc")
