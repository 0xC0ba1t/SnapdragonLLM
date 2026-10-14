import importlib.util
from pathlib import Path

import pytest
import torch

from snapdragon.export.safetensors_io import load_pretrained
from snapdragon.train.checkpointing import checkpoint_root, save_trainer_checkpoint

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_model.py"


def _cli():
    spec = importlib.util.spec_from_file_location("export_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ckpt(make_trainer, tiny_config, tmp_path, small_tokenizer):
    cfg = tiny_config()
    cfg.model.vocab_size = small_tokenizer.vocab_size
    cfg.train.out_dir = str(tmp_path / "runs")
    t = make_trainer(cfg)
    t.train_step()
    save_trainer_checkpoint(t)
    small_tokenizer.save(tmp_path / "tok.json")
    return t, checkpoint_root(t), tmp_path / "tok.json"


def test_exports_both_formats_and_the_weights_match_the_checkpoint(ckpt, tmp_path):
    trainer, root, tok = ckpt
    out = tmp_path / "release"
    written = _cli().main(["--checkpoint", str(root), "--out", str(out), "--tokenizer", str(tok),
                           "--format", "safetensors", "hf", "--dtype", "fp32"])
    assert [p.name for p in written] == ["snapdragon", "hf"]
    assert (out / "hf" / "config.json").exists()
    assert (out / "snapdragon" / "tokenizer.json").exists()
    reloaded = load_pretrained(out / "snapdragon")
    x = torch.randint(0, 64, (1, 8))
    trainer.raw_model.eval()
    with torch.no_grad():
        assert torch.allclose(reloaded(x), trainer.raw_model(x), atol=1e-5)


def test_hf_config_gets_the_special_token_ids(ckpt, tmp_path):
    import json

    _, root, tok = ckpt
    _cli().main(["--checkpoint", str(root), "--out", str(tmp_path / "r"), "--tokenizer", str(tok),
                 "--format", "hf"])
    cfg = json.loads((tmp_path / "r" / "hf" / "config.json").read_text())
    assert cfg["bos_token_id"] is not None
    assert cfg["eos_token_id"] == cfg["bos_token_id"] + 1
