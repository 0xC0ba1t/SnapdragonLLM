import pytest

from snapdragon.tokenizer.special_tokens import SPECIAL_TOKENS, special_token_ids


def test_specials_are_last_ids():
    ids = special_token_ids(32768)
    assert max(ids.values()) == 32767
    assert sorted(ids.values()) == list(range(32768 - len(SPECIAL_TOKENS), 32768))


def test_too_small_vocab():
    with pytest.raises(ValueError, match="too small"):
        special_token_ids(260)
