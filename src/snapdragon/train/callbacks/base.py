"""Callback protocol and ordered dispatch."""
from __future__ import annotations

from collections.abc import Iterable


class Callback:
    """Override any hook. ``priority`` orders dispatch (low runs first): metric producers use
    10, loggers 50 and checkpointing 90, so loggers always see metrics added by producers."""

    priority: int = 50

    def on_train_start(self, trainer) -> None: ...

    def on_step_end(self, trainer, metrics: dict) -> None:
        """Called after every optimizer step. ``metrics`` is mutable and shared by all callbacks."""

    def on_train_end(self, trainer) -> None: ...

    def on_exception(self, trainer, exc: BaseException) -> None: ...


class CallbackList:
    def __init__(self, callbacks: Iterable[Callback] = ()) -> None:
        self.callbacks = sorted(callbacks, key=lambda c: c.priority)  # stable: ties keep given order

    def add(self, callback: Callback) -> None:
        self.callbacks = sorted([*self.callbacks, callback], key=lambda c: c.priority)

    def on_train_start(self, trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_step_end(self, trainer, metrics: dict) -> None:
        for cb in self.callbacks:
            cb.on_step_end(trainer, metrics)

    def on_train_end(self, trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_exception(self, trainer, exc: BaseException) -> None:
        for cb in self.callbacks:
            cb.on_exception(trainer, exc)

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self) -> int:
        return len(self.callbacks)
