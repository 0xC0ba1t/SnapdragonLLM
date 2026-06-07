"""Tiny wall-clock timing helpers."""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class Timer:
    """Accumulating stopwatch. Use as a context manager or via start()/stop()."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            raise RuntimeError("Timer.stop() called before start()")
        dt = time.perf_counter() - self._t0
        self._t0 = None
        self.total += dt
        self.count += 1
        return dt

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


@contextmanager
def timed() -> Iterator[dict]:
    """with timed() as t: ...; then t['seconds'] holds the elapsed time."""
    out: dict = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - t0
