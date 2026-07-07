import pytest

from snapdragon.config.model import ModelConfig


def test_default_is_about_500m():
    n = ModelConfig().num_params()
    assert 480e6 < n < 560e6


def test_head_dim_derived():
    assert ModelConfig().head_dim == 64


def test_untied_embeddings_add_params():
    tied = ModelConfig(tie_embeddings=True).num_params()
    untied = ModelConfig(tie_embeddings=False).num_params()
    assert untied - tied == 32768 * 1280


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"d_model": 100, "n_heads": 3}, "not divisible by n_heads"),
        ({"n_heads": 20, "n_kv_heads": 3}, "n_kv_heads"),
        ({"act_ckpt": "sometimes"}, "act_ckpt"),
        ({"attn_impl": "magic"}, "attn_impl"),
        ({"d_model": 20, "n_heads": 4, "n_kv_heads": 2, "mlp_hidden": 8, "head_dim": 5}, "even"),
    ],
)
def test_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ModelConfig(**kwargs)


def test_suggest_mlp_hidden_rounds_up_to_multiple():
    from snapdragon.config.model import suggest_mlp_hidden

    assert suggest_mlp_hidden(1280) == 3456  # the 500M preset: 8/3 * 1280 = 3413.3 -> 27 * 128
    assert suggest_mlp_hidden(768) == 2048
    assert suggest_mlp_hidden(64, multiple=64) == 192
    assert suggest_mlp_hidden(1280) % 128 == 0
