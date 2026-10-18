"""Assemble a ready-to-run Trainer from a Config (what scripts/train.py calls)."""
from __future__ import annotations

from pathlib import Path

import torch

from snapdragon.checkpoint.model_io import load_weights_into
from snapdragon.config.io import dump_config
from snapdragon.config.schema import Config
from snapdragon.data.factory import build_train_loader, build_val_loader
from snapdragon.distributed.env import DistEnv, detect_env, device_for, init_distributed
from snapdragon.distributed.wrap import wrap_model
from snapdragon.model.compile import compile_blocks, should_compile
from snapdragon.model.factory import build_model, count_parameters
from snapdragon.optim.factory import build_optimizer
from snapdragon.optim.schedulers import LRSchedule
from snapdragon.sft.factory import build_sft_loader
from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.train.callbacks.base import Callback
from snapdragon.train.callbacks.benchmarks import BenchmarkCallback
from snapdragon.train.callbacks.checkpoint import CheckpointCallback, SignalHandler
from snapdragon.train.callbacks.loggers import ConsoleLogger, JsonlLogger
from snapdragon.train.callbacks.monitors import DivergenceGuard, MemoryCallback, WeightNormCallback
from snapdragon.train.callbacks.throughput import ThroughputCallback
from snapdragon.train.callbacks.validation import ValLossCallback
from snapdragon.train.checkpointing import maybe_resume
from snapdragon.train.trainer import Trainer
from snapdragon.utils.device import get_device
from snapdragon.utils.logging import get_logger
from snapdragon.utils.seed import seed_everything

log = get_logger(__name__)


def run_dir(cfg: Config) -> Path:
    return Path(cfg.train.out_dir) / cfg.train.run_name


def default_callbacks(cfg: Config) -> list[Callback]:
    return [
        SignalHandler(),
        ThroughputCallback(),
        MemoryCallback(),
        WeightNormCallback(every=100),
        DivergenceGuard(max_consecutive_skips=10),
        ConsoleLogger(every=cfg.train.log_every),
        JsonlLogger(run_dir(cfg), every=cfg.train.log_every),
        CheckpointCallback(every=cfg.train.save_every),
    ]


def build_trainer(
    cfg: Config,
    device: str | None = None,
    env: DistEnv | None = None,
    extra_callbacks: list[Callback] | None = None,
    train_loader=None,
    val_loader=None,
) -> Trainer:
    """Build model, optimizer, schedule, data loader and callbacks, then resume if configured.

    ``train_loader`` / ``val_loader`` can be injected (tests, custom data); otherwise they come from
    ``cfg.data``. A run without a validation split simply skips validation.
    """
    env = env or detect_env()
    init_distributed(env)
    rank, world_size = env.rank, env.world_size
    dev = device_for(env) if env.distributed and device is None else get_device(device)
    seed_everything(cfg.train.seed, rank)
    model = build_model(cfg).to(dev)
    log.info("model: %.1fM parameters on %s", count_parameters(model) / 1e6, dev)
    if cfg.train.init_from:
        load_weights_into(model, cfg.train.init_from, cfg.model)
        log.info("initialised weights from %s", cfg.train.init_from)
    if should_compile(cfg.train.compile):
        log.info("compiled %d blocks", compile_blocks(model))
    optimizer = build_optimizer(model, cfg.optim)
    schedule = LRSchedule(optimizer, cfg.optim.scheduler, cfg.train.total_steps)
    model = wrap_model(model, env, cfg.train.parallel)  # DDP shares the parameter objects, so the optimizer stays valid
    if train_loader is not None:
        loader = train_loader
    elif cfg.train.task == "sft":
        loader = build_sft_loader(cfg, rank, world_size)
    else:
        loader = build_train_loader(cfg, rank, world_size)
    callbacks = default_callbacks(cfg) + list(extra_callbacks or [])
    if val_loader is None and cfg.data.sources:
        try:
            val_loader = build_val_loader(cfg, rank, world_size)
        except (ValueError, FileNotFoundError) as e:
            log.warning("no validation loader (%s); skipping validation", e)
    if val_loader is not None:
        callbacks.append(ValLossCallback(val_loader, cfg.train.eval.every_steps))
    ev = cfg.train.eval
    if ev.benchmark_every > 0 and Path(cfg.data.tokenizer_path).exists():
        tok = FastTokenizer.from_file(cfg.data.tokenizer_path)
        callbacks.append(BenchmarkCallback(tok, ev.tasks, ev.benchmark_every, ev.limit))
    elif ev.benchmark_every > 0:
        log.warning("benchmark_every is set but %s does not exist; skipping benchmarks", cfg.data.tokenizer_path)
    trainer = Trainer(cfg, model, optimizer, schedule, loader, dev, callbacks, rank, world_size)
    if rank == 0:
        run_dir(cfg).mkdir(parents=True, exist_ok=True)
        dump_config(cfg, run_dir(cfg) / "config.yaml")
    maybe_resume(trainer)
    return trainer


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"{p.name} ({p.total_memory / 1e9:.1f} GB)"
    return str(device)
