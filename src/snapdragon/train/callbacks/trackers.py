"""Optional experiment trackers (TensorBoard, Weights & Biases). Imports are lazy and the callbacks
fail fast with an install hint, so the core package never depends on them."""
from __future__ import annotations

from pathlib import Path

from snapdragon.config.io import config_to_dict
from snapdragon.train.callbacks.base import Callback

_SKIP = {"step", "skipped"}


def _scalars(metrics: dict) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if k not in _SKIP and isinstance(v, (int, float))
            and not isinstance(v, bool)}


class TensorBoardLogger(Callback):
    priority = 60

    def __init__(self, log_dir: str | Path, every: int = 10) -> None:
        self.log_dir, self.every = str(log_dir), every
        self.writer = None

    def on_train_start(self, trainer) -> None:
        if not trainer.is_main:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError("TensorBoardLogger needs tensorboard: pip install -e .[logging]") from e
        self.writer = SummaryWriter(self.log_dir)

    def on_step_end(self, trainer, metrics: dict) -> None:
        if self.writer is not None and metrics["step"] % self.every == 0:
            for name, value in _scalars(metrics).items():
                self.writer.add_scalar(f"train/{name}", value, metrics["step"])

    def on_train_end(self, trainer) -> None:
        if self.writer is not None:
            self.writer.close()


class WandbLogger(Callback):
    priority = 60

    def __init__(self, project: str, run_name: str | None = None, every: int = 10, mode: str = "online") -> None:
        self.project, self.run_name, self.every, self.mode = project, run_name, every, mode
        self.run = None

    def on_train_start(self, trainer) -> None:
        if not trainer.is_main:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError("WandbLogger needs wandb: pip install -e .[logging]") from e
        self.run = wandb.init(project=self.project, name=self.run_name or trainer.cfg.train.run_name,
                              config=config_to_dict(trainer.cfg), mode=self.mode, resume="allow")

    def on_step_end(self, trainer, metrics: dict) -> None:
        if self.run is not None and metrics["step"] % self.every == 0:
            self.run.log(_scalars(metrics), step=metrics["step"])

    def on_train_end(self, trainer) -> None:
        if self.run is not None:
            self.run.finish()
