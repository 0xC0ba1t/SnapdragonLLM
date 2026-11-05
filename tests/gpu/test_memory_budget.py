"""The headline claim: the 500M presets train within their GPU's memory. `pytest -m gpu` on a 16 GB card."""
from pathlib import Path

import pytest
import torch

from snapdragon.config.io import load_config
from snapdragon.train.benchmark import run_benchmark
from snapdragon.train.memory import estimate_memory

pytestmark = pytest.mark.gpu

TRAIN = Path(__file__).resolve().parents[2] / "configs" / "train"


def _total_gb():
    return torch.cuda.get_device_properties(0).total_memory / 1e9


@pytest.mark.parametrize(("preset", "needs_gb"), [("pretrain-16gb", 15.0), ("pretrain-12gb", 11.5), ("pretrain-8gb", 7.5)])
def test_preset_peak_memory_fits_and_matches_the_estimate(preset, needs_gb):
    if _total_gb() < needs_gb:
        pytest.skip(f"needs a {needs_gb} GB GPU")
    cfg = load_config(TRAIN / f"{preset}.yaml", [f"train.global_batch_tokens={cfg_tokens(preset)}"])
    est = estimate_memory(cfg).total
    result = run_benchmark(cfg, steps=1, warmup=1, compile_model=False)
    assert result.peak_memory_gb < needs_gb
    assert result.peak_memory_gb < 1.3 * est  # the estimate is conservative but not wildly off
    assert result.peak_memory_gb > 0.5 * est


def cfg_tokens(preset):
    """Two micro-batches per step keeps the benchmark short while exercising accumulation."""
    cfg = load_config(TRAIN / f"{preset}.yaml")
    return 2 * cfg.train.micro_batch_size * cfg.train.seq_len
