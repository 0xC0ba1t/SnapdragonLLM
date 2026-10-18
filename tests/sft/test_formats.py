import pytest

from snapdragon.sft.formats import from_alpaca, from_messages, from_sharegpt, normalise


def test_alpaca_joins_instruction_and_input():
    out = from_alpaca({"instruction": "Translate", "input": "hello", "output": "bonjour"})
    assert out == [{"role": "user", "content": "Translate\n\nhello"}, {"role": "assistant", "content": "bonjour"}]
    assert from_alpaca({"instruction": "Say hi", "output": "hi"})[0]["content"] == "Say hi"


def test_sharegpt_roles_are_normalised():
    out = from_sharegpt({"conversations": [{"from": "system", "value": "be nice"}, {"from": "human", "value": "q"},
                                           {"from": "gpt", "value": "a"}]})
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    with pytest.raises(ValueError, match="unknown sharegpt speaker"):
        from_sharegpt({"conversations": [{"from": "alien", "value": "x"}]})


def test_messages_aliases_and_strip():
    out = from_messages({"messages": [{"role": "Human", "content": " hi "}, {"role": "bot", "content": "yo"}]})
    assert out == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]


def test_normalise_drops_unusable_conversations():
    rows = [
        {"instruction": "ok", "output": "fine"},
        {"instruction": "no answer", "output": "  "},
        {"instruction": "ok2", "output": "fine2"},
    ]
    assert len(list(normalise(rows, "alpaca"))) == 2


def test_normalise_requires_user_first_assistant_last_and_alternation():
    bad = [
        {"conversations": [{"from": "gpt", "value": "a"}, {"from": "human", "value": "q"}]},
        {"conversations": [{"from": "human", "value": "q"}]},
        {"conversations": [{"from": "human", "value": "q1"}, {"from": "human", "value": "q2"},
                           {"from": "gpt", "value": "a"}]},
    ]
    good = [{"conversations": [{"from": "system", "value": "s"}, {"from": "human", "value": "q"},
                               {"from": "gpt", "value": "a"}, {"from": "human", "value": "q2"},
                               {"from": "gpt", "value": "a2"}]}]
    assert list(normalise(bad, "sharegpt")) == []
    assert len(list(normalise(good, "sharegpt"))) == 1


def test_unknown_format():
    with pytest.raises(ValueError, match="unknown format"):
        list(normalise([], "csv"))
