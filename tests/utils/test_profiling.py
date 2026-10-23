import json

import torch

from snapdragon.utils.profiling import memory_history, profile_run


def test_profile_run_writes_a_chrome_trace_and_an_ops_table(tmp_path):
    a = torch.randn(64, 64)
    with profile_run(tmp_path / "prof", wait=0, warmup=1, active=2) as prof:
        for _ in range(4):
            (a @ a).sum()
            prof.step()
    trace = json.loads((tmp_path / "prof" / "trace.json").read_text(encoding="utf-8"))
    assert "traceEvents" in trace
    table = (tmp_path / "prof" / "top_ops.txt").read_text(encoding="utf-8")
    assert "aten::" in table


def test_profile_run_creates_missing_directories(tmp_path):
    with profile_run(tmp_path / "a" / "b", wait=0, warmup=0, active=1) as prof:
        torch.ones(2).sum()
        prof.step()
    assert (tmp_path / "a" / "b" / "trace.json").exists()


def test_memory_history_is_a_safe_noop_without_cuda(tmp_path):
    with memory_history(tmp_path / "snap.pickle") as active:
        assert active is torch.cuda.is_available()
    if not torch.cuda.is_available():
        assert not (tmp_path / "snap.pickle").exists()
