import pytest
import torch

from snapdragon.checkpoint.model_io import load_model_from_checkpoint, resolve_checkpoint
from snapdragon.train.checkpointing import checkpoint_root, save_trainer_checkpoint


@pytest.fixture
def trained(make_trainer, tiny_config, tmp_path):
    cfg = tiny_config()
    cfg.train.out_dir = str(tmp_path)
    t = make_trainer(cfg)
    for _ in range(3):
        t.train_step()
    save_trainer_checkpoint(t)
    return t


def test_loaded_model_reproduces_the_trained_models_outputs(trained):
    model, cfg = load_model_from_checkpoint(checkpoint_root(trained))
    assert cfg.model == trained.cfg.model
    x = torch.randint(0, 64, (2, 8))
    trained.raw_model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), trained.raw_model(x), atol=1e-6)
    assert not model.training


def test_accepts_a_specific_step_directory(trained):
    step_dir = resolve_checkpoint(checkpoint_root(trained))
    assert step_dir.name == "step_00000003"
    model, _ = load_model_from_checkpoint(step_dir)
    assert next(model.parameters()).device.type == "cpu"


def test_no_checkpoint_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no complete checkpoint"):
        resolve_checkpoint(tmp_path)


def test_load_weights_into_copies_weights_only(trained):
    import dataclasses

    from snapdragon.checkpoint.model_io import load_weights_into
    from snapdragon.model.factory import build_model

    fresh = build_model(dataclasses.replace(trained.cfg.model, init_seed=777))
    assert not torch.equal(fresh.transformer.embed.weight, trained.raw_model.transformer.embed.weight)
    load_weights_into(fresh, checkpoint_root(trained))
    assert torch.equal(fresh.transformer.embed.weight, trained.raw_model.transformer.embed.weight)


def test_load_weights_into_rejects_a_different_architecture(trained):
    import dataclasses

    from snapdragon.checkpoint.model_io import load_weights_into
    from snapdragon.model.factory import build_model

    other = build_model(dataclasses.replace(trained.cfg.model, n_layers=3))
    with pytest.raises(ValueError, match="architecture differs.*n_layers"):
        load_weights_into(other, checkpoint_root(trained))
