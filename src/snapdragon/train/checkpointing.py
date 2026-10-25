"""What goes into a training checkpoint, and how to put it back."""
from __future__ import annotations

from pathlib import Path

import snapdragon
from snapdragon.checkpoint.io import (
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from snapdragon.config.io import config_to_dict
from snapdragon.utils.logging import get_logger
from snapdragon.utils.seed import get_rng_state, set_rng_state

log = get_logger(__name__)


def checkpoint_root(trainer) -> Path:
    return Path(trainer.cfg.train.out_dir) / trainer.cfg.train.run_name / "checkpoints"


def trainer_components(trainer) -> dict:
    """Everything needed to continue a run bit-for-bit: weights, optimizer (incl. int8 state), LR
    schedule position, data position, RNG streams, progress counters, plus the config for provenance."""
    extras = {f"extra_{name}": obj.state_dict() for name, obj in trainer.extra_state.items()}
    return {
        **extras,
        "model": trainer.raw_model.state_dict(),
        "optim": trainer.optimizer.state_dict(),
        "schedule": trainer.schedule.state_dict(),
        "loader": trainer.train_loader.state_dict(),
        "trainer_state": trainer.state.state_dict(),
        "rng": get_rng_state(),
        "meta": {"config": config_to_dict(trainer.cfg), "version": snapdragon.__version__},
    }


def save_trainer_checkpoint(trainer) -> Path:
    root = checkpoint_root(trainer)
    path = save_checkpoint(root, trainer.state.step, trainer_components(trainer))
    prune_checkpoints(root, trainer.cfg.train.keep_last_n)
    log.info("saved checkpoint %s", path)
    return path


def restore_trainer(trainer, components: dict, restore_rng: bool = True) -> None:
    """Load a checkpoint's components into a freshly built trainer."""
    trainer.raw_model.load_state_dict(components["model"])
    trainer.optimizer.load_state_dict(components["optim"])
    trainer.schedule.load_state_dict(components["schedule"])
    trainer.state.load_state_dict(components["trainer_state"])
    trainer.train_loader.load_state_dict(components["loader"])
    for name, obj in trainer.extra_state.items():
        if f"extra_{name}" in components:
            obj.load_state_dict(components[f"extra_{name}"])
    if restore_rng:
        set_rng_state(components["rng"])


def resolve_resume_path(trainer, resume: str) -> Path | None:
    """``resume``: "none" -> never; "auto" -> newest checkpoint of this run if any; else a path."""
    if resume == "none":
        return None
    if resume == "auto":
        return latest_checkpoint(checkpoint_root(trainer))
    return Path(resume)


def maybe_resume(trainer) -> bool:
    path = resolve_resume_path(trainer, trainer.cfg.train.resume)
    if path is None:
        return False
    restore_trainer(trainer, load_checkpoint(path, map_location=trainer.device))
    log.info("resumed from %s at step %d", path, trainer.state.step)
    return True
