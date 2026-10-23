import importlib.util
from pathlib import Path

import pytest
import torch

from snapdragon.train.benchmark import BenchmarkResult, RandomLoader, run_benchmark

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"


def test_random_loader_shapes_and_shift():
    loader = RandomLoader(3, 8, 50)
    b = next(loader)
    assert b.x.shape == b.y.shape == (3, 8)
    assert b.x.max() < 50
    assert loader.batch_index == 1


def test_random_loader_is_seeded():
    assert torch.equal(next(RandomLoader(2, 4, 99, seed=1)).x, next(RandomLoader(2, 4, 99, seed=1)).x)
    assert not torch.equal(next(RandomLoader(2, 4, 99, seed=1)).x, next(RandomLoader(2, 4, 99, seed=2)).x)


def test_run_benchmark_on_a_tiny_cpu_config(tiny_config):
    r = run_benchmark(tiny_config(), steps=2, warmup=1, device="cpu", compile_model=False)
    assert r.tokens_per_step == 64
    assert r.tokens_per_sec > 0
    assert r.seconds_per_step > 0
    assert r.mfu is None  # no known peak for a CPU
    assert r.peak_memory_gb is None
    assert "tok/s" in r.summary()


def test_summary_includes_gpu_fields_when_present():
    s = BenchmarkResult(5, 2.0, 12_000.0, 0.31, 9.4, 24_000).summary()
    assert "12.00k tok/s" in s
    assert "MFU 31.0%" in s
    assert "peak 9.4 GB" in s


def test_benchmark_cli_prints_prediction_and_duration(capsys):
    spec = importlib.util.spec_from_file_location("bench_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = SCRIPT.parents[1]
    result = mod.main(["--config", str(root / "configs" / "train" / "debug-cpu.yaml"), "--steps", "1", "--warmup", "1",
                       "--device", "cpu", "--no-compile", "train.micro_batch_size=2", "train.global_batch_tokens=512",
                       "train.seq_len=128", "train.seq_len_warmup_tokens=0"])
    out = capsys.readouterr().out
    assert "predicted peak" in out
    assert "days at this speed" in out
    assert result.tokens_per_sec > 0
    with pytest.raises(SystemExit):
        mod.main([])
