import importlib.util
from pathlib import Path

from snapdragon.train.checkpointing import checkpoint_root, save_trainer_checkpoint

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inspect_model.py"


def _cli():
    spec = importlib.util.spec_from_file_location("inspect_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_mode_reports_500m_without_allocating_it():
    root = SCRIPT.parents[1]
    text = _cli().main(["--config", str(root / "configs" / "model" / "snapdragon-500m.yaml")])
    assert "528." in text  # 528.3M parameters
    assert "embeddings" in text
    assert "GFLOPs/token" in text
    assert "heads=20/5" in text


def test_checkpoint_mode_shows_training_progress(make_trainer, tiny_config, tmp_path):
    cfg = tiny_config()
    cfg.train.out_dir = str(tmp_path)
    t = make_trainer(cfg)
    for _ in range(2):
        t.train_step()
    save_trainer_checkpoint(t)
    text = _cli().main(["--checkpoint", str(checkpoint_root(t))])
    assert "step 2" in text
    assert "step_00000002" in text
