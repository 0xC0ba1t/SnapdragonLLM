import copy

import torch
import yaml

from snapdragon.checkpoint.io import list_checkpoints
from snapdragon.config.data import SourceConfig
from snapdragon.data.manifest import DatasetManifest, ShardInfo
from snapdragon.data.shards import ShardWriter
from snapdragon.train.checkpointing import checkpoint_root
from snapdragon.train.setup import build_trainer, default_callbacks, run_dir


def _write_learnable_dataset(path, n=20_000, vocab=64):
    """Token ids stay inside the tiny model's vocab; next token is a function of the previous one."""
    m = DatasetManifest(vocab_size=vocab, eos_id=vocab - 1)
    for split, size in (("train", n), ("val", 1000)):
        with ShardWriter(path, split, 10**9, eos_id=vocab - 1) as w:
            w.add_document([(i * 5 + 3) % (vocab - 1) for i in range(size)])
        m.shards += [ShardInfo(s, t, d, split) for s, t, d in w.files]
    m.save(path)


def _cfg(tiny_config, tmp_path):
    cfg = tiny_config()
    _write_learnable_dataset(tmp_path / "data")
    cfg.data.sources = [SourceConfig("main", str(tmp_path / "data"))]
    cfg.data.pin_memory = False
    cfg.data.prefetch_batches = 2
    cfg.train.out_dir = str(tmp_path / "runs")
    cfg.train.compile = "false"
    cfg.train.save_every = 10
    cfg.train.log_every = 5
    cfg.train.max_tokens = cfg.train.global_batch_tokens * 12
    return cfg


def test_default_callbacks_are_ordered_producers_first(tiny_config):
    cbs = default_callbacks(tiny_config())
    names = [type(c).__name__ for c in sorted(cbs, key=lambda c: c.priority)]
    assert names.index("ThroughputCallback") < names.index("ConsoleLogger")
    assert names[-1] == "CheckpointCallback"
    assert names[0] == "SignalHandler"


def test_end_to_end_run_from_a_config(tiny_config, tmp_path):
    cfg = _cfg(tiny_config, tmp_path)
    trainer = build_trainer(cfg, device="cpu")
    try:
        state = trainer.fit()
    finally:
        trainer.train_loader.close()
    assert state.step == 12
    assert (run_dir(cfg) / "metrics.jsonl").exists()
    saved = yaml.safe_load((run_dir(cfg) / "config.yaml").read_text())
    assert saved["model"]["n_layers"] == 2
    assert [s for s, _ in list_checkpoints(checkpoint_root(trainer))] == [10, 12]


def test_validation_runs_when_the_dataset_has_a_val_split(tiny_config, tmp_path):
    import json

    cfg = _cfg(tiny_config, tmp_path)
    cfg.train.eval.every_steps = 6
    cfg.train.log_every = 1  # the jsonl logger only records logged steps
    cfg.train.eval.val_batches = 2
    trainer = build_trainer(cfg, device="cpu")
    try:
        trainer.fit()
    finally:
        trainer.train_loader.close()
    rows = [json.loads(line) for line in (run_dir(cfg) / "metrics.jsonl").read_text().splitlines()]
    with_val = [r["step"] for r in rows if "val_loss" in r]
    assert with_val == [6, 12]
    assert trainer.state.best_val_loss < float("inf")


def test_rerun_resumes_from_the_last_checkpoint(tiny_config, tmp_path):
    cfg = _cfg(tiny_config, tmp_path)
    first = build_trainer(cfg, device="cpu")
    try:
        first.fit()
    finally:
        first.train_loader.close()

    longer = copy.deepcopy(cfg)
    longer.train.max_tokens = cfg.train.global_batch_tokens * 15
    second = build_trainer(longer, device="cpu")
    try:
        assert second.state.step == 12  # picked up where the first run ended
        assert second.state.tokens_seen == first.state.tokens_seen
        end = second.fit()
    finally:
        second.train_loader.close()
    assert end.step == 15
    assert torch.isfinite(torch.tensor(end.loss_ema))


def test_explicit_single_process_env_trains_without_a_process_group(tiny_config, tmp_path):
    import torch.distributed as dist

    from snapdragon.distributed.env import DistEnv

    cfg = _cfg(tiny_config, tmp_path)
    trainer = build_trainer(cfg, device="cpu", env=DistEnv())
    try:
        assert trainer.world_size == 1
        assert trainer.model is trainer.raw_model  # not wrapped in DDP
        assert not dist.is_initialized()
    finally:
        trainer.train_loader.close()
