import snapdragon


def test_version_string():
    assert snapdragon.__version__.count(".") == 2
