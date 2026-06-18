import pytest

from snapdragon.tokenizer.pretokenize import pretokenize


def test_simple_sentence():
    assert pretokenize("Hello world!") == ["Hello", " world", "!"]


def test_digits_split_in_groups_of_three():
    assert pretokenize("12345") == ["123", "45"]


def test_contractions():
    assert pretokenize("I'm sure it's fine") == ["I", "'m", " sure", " it", "'s", " fine"]


@pytest.mark.parametrize("text", ["", "   ", "a\n\nb", "naïve café 你好", "tab\tseparated", "emoji 🙂!"])
def test_pretokenize_is_lossless(text):
    assert "".join(pretokenize(text)) == text


def test_curly_apostrophe_contractions():
    assert pretokenize("don’t stop") == ["don", "’t", " stop"]
