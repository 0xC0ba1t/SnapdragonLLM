import json

import pytest
import torch
from safetensors import safe_open

from snapdragon.config.model import ModelConfig
from snapdragon.export.safetensors_io import CONFIG, WEIGHTS, load_pretrained, save_pretrained
from snapdragon.model.factory import build_model

CFG = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=16)


def test_fp32_roundtrip_is_exact(tmp_path):
    torch.manual_seed(0)
    m = build_model(CFG).eval()
    save_pretrained(m, tmp_path, dtype=None)
    again = load_pretrained(tmp_path)
    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        assert torch.equal(m(x), again(x))


def test_bf16_export_halves_the_size_and_stays_close(tmp_path):
    torch.manual_seed(0)
    m = build_model(CFG).eval()
    save_pretrained(m, tmp_path / "fp32", dtype=None)
    save_pretrained(m, tmp_path / "bf16")
    big = (tmp_path / "fp32" / WEIGHTS).stat().st_size
    small = (tmp_path / "bf16" / WEIGHTS).stat().st_size
    assert small < 0.55 * big
    again = load_pretrained(tmp_path / "bf16")
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        assert torch.allclose(m(x), again(x), atol=0.05)


def test_tied_weights_are_stored_once(tmp_path):
    save_pretrained(build_model(CFG), tmp_path, dtype=None)
    with safe_open(str(tmp_path / WEIGHTS), "pt") as f:
        keys = list(f.keys())
    assert "transformer.embed.weight" in keys
    assert not any("lm_head" in k for k in keys)


def test_config_is_self_describing_json(tmp_path):
    save_pretrained(build_model(CFG), tmp_path)
    cfg = json.loads((tmp_path / CONFIG).read_text())
    assert cfg["architecture"] == "snapdragon"
    assert cfg["d_model"] == 32
    assert cfg["n_kv_heads"] == 2


def test_tokenizer_is_copied_along(tmp_path, small_tokenizer):
    small_tokenizer.save(tmp_path / "src.json")
    out = save_pretrained(build_model(CFG), tmp_path / "out", tokenizer_path=tmp_path / "src.json")
    assert (out / "tokenizer.json").read_bytes() == (tmp_path / "src.json").read_bytes()


def test_untied_model_roundtrips(tmp_path):
    import dataclasses

    cfg = dataclasses.replace(CFG, tie_embeddings=False)
    torch.manual_seed(1)
    m = build_model(cfg).eval()
    save_pretrained(m, tmp_path, dtype=None)
    again = load_pretrained(tmp_path)
    x = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        assert torch.equal(m(x), again(x))


def test_foreign_config_is_rejected(tmp_path):
    save_pretrained(build_model(CFG), tmp_path)
    cfg = json.loads((tmp_path / CONFIG).read_text())
    cfg["architecture"] = "llama"
    (tmp_path / CONFIG).write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="not a snapdragon config"):
        load_pretrained(tmp_path)
