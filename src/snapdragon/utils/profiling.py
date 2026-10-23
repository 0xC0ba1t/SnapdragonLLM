"""torch.profiler and CUDA memory-history helpers."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule


@contextlib.contextmanager
def profile_run(out_dir: str | Path, wait: int = 1, warmup: int = 1, active: int = 2,
                top: int = 15) -> Iterator[profile]:
    """Profile a training loop: call ``prof.step()`` once per iteration.

    Skips ``wait`` iterations, warms up for ``warmup`` and records ``active``. On exit writes
    ``trace.json`` (open in chrome://tracing or ui.perfetto.dev) and ``top_ops.txt`` (the ``top``
    most expensive ops by self time).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    prof = profile(
        activities=activities, schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        record_shapes=True, profile_memory=True, with_stack=False,
    )
    with prof:
        yield prof
    prof.export_chrome_trace(str(out / "trace.json"))
    table = prof.key_averages().table(sort_by=sort_key, row_limit=top)
    (out / "top_ops.txt").write_text(table, encoding="utf-8")


@contextlib.contextmanager
def memory_history(out_path: str | Path, max_entries: int = 100_000) -> Iterator[bool]:
    """Record every CUDA allocation/free and dump a snapshot for pytorch.org/memory_viz on exit.

    Yields whether recording is active (False on machines without CUDA, where this is a no-op).
    """
    if not torch.cuda.is_available():
        yield False
        return
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield True
    finally:
        torch.cuda.memory._dump_snapshot(str(out_path))
        torch.cuda.memory._record_memory_history(enabled=None)
