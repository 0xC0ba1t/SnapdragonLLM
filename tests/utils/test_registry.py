import pytest

from snapdragon.utils.registry import Registry


def test_register_and_get():
    reg: Registry = Registry("widget")

    @reg.register("a")
    def make_a():
        return 1

    assert reg.get("a")() == 1
    assert "a" in reg
    assert reg.names() == ["a"]
    assert len(reg) == 1


def test_duplicate_registration_fails():
    reg: Registry = Registry("widget")
    reg.register("a")(object())
    with pytest.raises(KeyError, match="already registered"):
        reg.register("a")(object())


def test_unknown_lists_available():
    reg: Registry = Registry("widget")
    reg.register("b")(object())
    with pytest.raises(KeyError, match=r"available: \['b'\]"):
        reg.get("zzz")
