from snapdragon.config.model import ModelConfig
from snapdragon.model.factory import build_model, count_parameters
from snapdragon.model.summary import format_summary, parameter_breakdown

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def test_groups_sum_to_total():
    m = build_model(CFG)
    groups = parameter_breakdown(m)
    assert sum(groups.values()) == count_parameters(m)
    assert groups["other"] == 0


def test_known_sizes():
    groups = parameter_breakdown(build_model(CFG))
    assert groups["embeddings"] == 64 * 32
    assert groups["mlp"] == 2 * 3 * 32 * 64
    d, hd = 32, 8
    assert groups["attention"] == 2 * (d * 4 * hd + 2 * d * 2 * hd + 4 * hd * d)
    assert groups["norms"] == 2 * (2 * d + 2 * hd) + d


def test_summary_text_lists_total():
    text = format_summary(build_model(CFG))
    assert "embeddings" in text
    assert text.splitlines()[-1].startswith("total")
