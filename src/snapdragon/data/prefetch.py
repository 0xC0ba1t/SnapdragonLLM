"""Background-thread prefetching so batch construction overlaps with GPU compute."""
from __future__ import annotations

import queue
import threading

from snapdragon.data.batches import BatchBuilder
from snapdragon.data.loader import Batch

_END = object()


class PrefetchLoader:
    """Like TrainLoader, but a worker thread keeps ``depth`` batches ready.

    ``batch_index`` counts batches *handed to the consumer*, not prefetched ones, so
    ``state_dict()`` is exactly the resume point even though the worker is ahead.
    """

    def __init__(self, builder: BatchBuilder, start_batch: int = 0, depth: int = 4) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.builder = builder
        self.batch_index = start_batch
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._work, args=(start_batch,), daemon=True)
        self._thread.start()

    def _work(self, n: int) -> None:
        try:
            while not self._stop.is_set():
                x, y = self.builder.get_batch(n)
                item = Batch(x, y)
                while not self._stop.is_set():
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                n += 1
        except BaseException as e:  # surfaced to the consumer on the next __next__
            self._queue.put(e)
        finally:
            self._queue.put(_END)

    def __iter__(self) -> PrefetchLoader:
        return self

    def __next__(self) -> Batch:
        item = self._queue.get()
        if item is _END:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        self.batch_index += 1
        return item

    def state_dict(self) -> dict:
        return {"batch_index": self.batch_index}

    def close(self) -> None:
        self._stop.set()
        # unblock a worker waiting on a full queue
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)

    def __enter__(self) -> PrefetchLoader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
