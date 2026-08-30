import importlib.util
from pathlib import Path

import pytest

from snapdragon.config.io import load_config
from snapdragon.train.memory import estimate_memory

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "configs" / "train"
PRESETS = sorted(p for p in TRAIN.glob("*.yaml"))


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.stem)
def test_every_train_preset_loads_and_has_a_valid_batch_layout(path):
    cfg = load_config(path)
    assert cfg.train.grad_accum_steps() >= 1
    assert cfg.train.seq_len == cfg.data.seq_len <= cfg.model.max_seq_len


@pytest.mark.parametrize(
    ("name", "budget_gb"),
    [("pretrain-16gb", 15.0), ("pretrain-12gb", 11.5), ("pretrain-8gb", 7.5)],
)
def test_500m_presets_fit_their_stated_gpu(name, budget_gb):
    cfg = load_config(TRAIN / f"{name}.yaml")
    assert 480e6 < cfg.model.num_params() < 560e6
    assert estimate_memory(cfg).total < budget_gb


def test_presets_share_one_global_batch_so_results_are_comparable():
    batches = {load_config(TRAIN / f"pretrain-{s}.yaml").train.global_batch_tokens for s in ("16gb", "12gb", "8gb")}
    assert batches == {524_288}


def test_8gb_preset_offloads_optimizer_state():
    cfg = load_config(TRAIN / "pretrain-8gb.yaml")
    assert cfg.optim.offload_states
    assert cfg.model.act_ckpt == "full"


def test_estimate_memory_cli_prints_a_table(capsys):
    spec = importlib.util.spec_from_file_location("estimate_cli", ROOT / "scripts" / "estimate_memory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    total = mod.main(["--config", str(TRAIN / "pretrain-16gb.yaml"), "--budget-gb", "15"])
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert "GFLOPs/token" in out
    assert total < 15
    with pytest.raises(SystemExit, match="exceeds"):
        mod.main(["--config", str(TRAIN / "pretrain-16gb.yaml"), "--budget-gb", "5"])
