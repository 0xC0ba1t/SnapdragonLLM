import pytest

from snapdragon.data.filters import QualityFilter

GOOD = "The committee reviewed several proposals about renewable energy and decided to fund two of them. " * 4


@pytest.fixture
def f():
    return QualityFilter()


def test_good_text_is_kept(f):
    assert f.reason_to_drop(GOOD) is None
    assert f.keep(GOOD)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("short", "too_short"),
        ("1234 5678 9012 " * 40, "low_alpha"),
        ("a b c d e f g h i j k l m n o p q r s t " * 20, "word_length"),
        ("\n".join(["buy now cheap pills online store"] * 20), "repeated_lines"),
    ],
)
def test_drop_reasons(f, text, reason):
    assert f.reason_to_drop(text) == reason


def test_thresholds_are_configurable():
    assert QualityFilter(min_chars=1).keep("tiny but fine")


def test_too_long_uses_max_chars():
    assert QualityFilter(max_chars=50).reason_to_drop(GOOD) == "too_long"
