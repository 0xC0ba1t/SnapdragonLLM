import json

import pytest

from snapdragon.sft.data import SFTExample, read_messages, tokenize_chats

CHAT = [
    {"role": "user", "content": "abc def"},
    {"role": "assistant", "content": "ghi jab"},
]


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_reads_valid_jsonl(tmp_path):
    p = tmp_path / "sft.jsonl"
    _write(p, [{"messages": CHAT}, {"messages": CHAT[:1] + CHAT}])
    assert [len(m) for m in read_messages(p)] == [2, 3]


@pytest.mark.parametrize(
    "row",
    [{"nope": 1}, {"messages": []}, {"messages": [{"role": "robot", "content": "x"}]},
     {"messages": [{"role": "user", "content": 5}]}],
)
def test_bad_rows_name_the_file_and_line(tmp_path, row):
    p = tmp_path / "bad.jsonl"
    _write(p, [{"messages": CHAT}, row])
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        list(read_messages(p))


def test_mask_covers_only_the_assistant_turn(small_tokenizer):
    (ex,) = tokenize_chats(small_tokenizer, [CHAT])
    trained = [i for i, m in zip(ex.ids, ex.mask) if m]
    assert small_tokenizer.decode(trained) == "ghi jab<|end_turn|>"


def test_chats_without_assistant_text_or_too_long_are_skipped(small_tokenizer):
    user_only = [{"role": "user", "content": "abc"}]
    long_chat = [{"role": "user", "content": "abc " * 200}, {"role": "assistant", "content": "def"}]
    out = list(tokenize_chats(small_tokenizer, [user_only, long_chat, CHAT], max_len=64))
    assert len(out) == 1


def test_example_lengths_must_agree():
    with pytest.raises(ValueError, match="same length"):
        SFTExample([1, 2], [1])
