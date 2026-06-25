"""Turn a stream of documents into tokenized train/val shards plus a manifest."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from snapdragon.data.dedup import ExactDedup
from snapdragon.data.filters import QualityFilter
from snapdragon.data.manifest import DatasetManifest, ShardInfo
from snapdragon.data.shards import ShardWriter
from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.utils.logging import get_logger

log = get_logger(__name__)


def _batched(texts: Iterable[str], n: int):
    batch: list[str] = []
    for t in texts:
        batch.append(t)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def tokenize_to_shards(
    texts: Iterable[str],
    tok: FastTokenizer,
    out_dir: str | Path,
    shard_tokens: int = 100_000_000,
    val_tokens: int = 10_000_000,
    max_tokens: int | None = None,
    batch_size: int = 512,
    quality_filter: QualityFilter | None = None,
    dedup: ExactDedup | None = None,
    stats: Counter | None = None,
) -> DatasetManifest:
    """Write ``val_*`` shards first (until ``val_tokens``), then ``train_*`` shards.

    Stops after ``max_tokens`` total tokens if given. Documents rejected by ``quality_filter`` or
    seen before by ``dedup`` are skipped and counted in ``stats`` (keys: kept, dup, or the filter's
    drop reason). Returns (and saves) the manifest.
    """
    stats = stats if stats is not None else Counter()
    out_dir = Path(out_dir)
    val = ShardWriter(out_dir, "val", shard_tokens, tok.eos_id)
    train = ShardWriter(out_dir, "train", shard_tokens, tok.eos_id)
    val_done = val_tokens <= 0
    seen_val = seen_total = 0
    stop = False
    def accepted(texts_in):
        for t in texts_in:
            reason = quality_filter.reason_to_drop(t) if quality_filter else None
            if reason:
                stats[reason] += 1
            elif dedup is not None and dedup.is_duplicate(t):
                stats["dup"] += 1
            else:
                stats["kept"] += 1
                yield t

    for batch in _batched(accepted(texts), batch_size):
        for ids in tok.encode_batch(batch):
            n = len(ids) + 1
            if not val_done:
                val.add_document(ids)
                seen_val += n
                val_done = seen_val >= val_tokens
            else:
                train.add_document(ids)
            seen_total += n
            if max_tokens is not None and seen_total >= max_tokens:
                stop = True
                break
        if stop:
            break
    manifest = DatasetManifest(vocab_size=tok.vocab_size, eos_id=tok.eos_id)
    for split, writer in (("val", val), ("train", train)):
        for stem, tokens, docs in writer.close():
            manifest.shards.append(ShardInfo(stem, tokens, docs, split))
    manifest.save(out_dir)
    log.info("wrote %d tokens in %d shards", manifest.total_tokens(), len(manifest.shards))
    return manifest
