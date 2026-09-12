"""Periodic and emergency checkpointing, and graceful shutdown on signals."""
from __future__ import annotations

import signal

from snapdragon.distributed.env import barrier
from snapdragon.train.callbacks.base import Callback
from snapdragon.train.checkpointing import save_trainer_checkpoint
from snapdragon.utils.logging import get_logger

log = get_logger(__name__)


class CheckpointCallback(Callback):
    """Saves every ``every`` steps, at the end of training, and when training is interrupted.

    Runs last (priority 90) so the checkpoint contains whatever earlier callbacks recorded.
    Only the main process writes.
    """

    priority = 90

    def __init__(self, every: int) -> None:
        if every < 1:
            raise ValueError("every must be at least 1")
        self.every = every
        self._last_saved_step = -1

    def _save(self, trainer, sync: bool = True) -> None:
        if trainer.is_main and trainer.state.step != self._last_saved_step:
            save_trainer_checkpoint(trainer)
        self._last_saved_step = trainer.state.step
        if sync:
            barrier()  # other ranks wait for rank 0's write instead of racing ahead into a collective timeout

    def on_step_end(self, trainer, metrics: dict) -> None:
        if metrics["step"] % self.every == 0:
            self._save(trainer)

    def on_train_end(self, trainer) -> None:
        self._save(trainer)

    def on_exception(self, trainer, exc: BaseException) -> None:
        if isinstance(exc, KeyboardInterrupt) and trainer.state.step > 0:
            log.warning("interrupted: saving an emergency checkpoint")
            self._save(trainer, sync=False)  # peers may not be at a barrier; never block while failing


class SignalHandler(Callback):
    """SIGINT/SIGTERM request a clean stop after the current step instead of killing the process,
    so the final checkpoint is written (cluster pre-emption sends SIGTERM)."""

    priority = 5

    def __init__(self, signals: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)) -> None:
        self.signals = signals
        self._previous: dict[int, object] = {}

    def on_train_start(self, trainer) -> None:
        def handler(signum, frame):
            log.warning("received signal %s: stopping after this step", signum)
            trainer.stop_requested = True

        for s in self.signals:
            self._previous[s] = signal.signal(s, handler)

    def on_train_end(self, trainer) -> None:
        for s, prev in self._previous.items():
            signal.signal(s, prev)
        self._previous.clear()

    def on_exception(self, trainer, exc: BaseException) -> None:
        self.on_train_end(trainer)
