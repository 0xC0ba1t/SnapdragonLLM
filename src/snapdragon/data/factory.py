"""Build the train/val loaders described by a Config."""
from __future__ import annotations

from snapdragon.config.schema import Config
from snapdragon.data.batches import BatchBuilder
from snapdragon.data.memmap_dataset import TokenDataset
from snapdragon.data.prefetch import PrefetchLoader
from snapdragon.data.val import ValLoader


def build_train_loader(
    cfg: Config, rank: int = 0, world_size: int = 1, start_batch: int = 0, seq_len: int | None = None
) -> PrefetchLoader:
    if not cfg.data.sources:
        raise ValueError("data.sources is empty; point the config at a prepared dataset")
    datasets = [TokenDataset(s.path, "train") for s in cfg.data.sources]
    weights = [s.weight for s in cfg.data.sources]
    builder = BatchBuilder(
        datasets,
        weights,
        seq_len=seq_len or cfg.train.seq_len,
        micro_batch_size=cfg.train.micro_batch_size,
        seed=cfg.data.seed,
        rank=rank,
        world_size=world_size,
    )
    return PrefetchLoader(
        builder, start_batch=start_batch, depth=cfg.data.prefetch_batches, pin_memory=cfg.data.pin_memory
    )


def build_val_loader(cfg: Config, rank: int = 0, world_size: int = 1) -> ValLoader:
    """Validation always uses the first source's val split."""
    if not cfg.data.sources:
        raise ValueError("data.sources is empty; point the config at a prepared dataset")
    ds = TokenDataset(cfg.data.sources[0].path, "val")
    return ValLoader(
        ds,
        cfg.train.seq_len,
        cfg.train.micro_batch_size,
        max_batches=cfg.train.eval.val_batches,
        rank=rank,
        world_size=world_size,
    )
