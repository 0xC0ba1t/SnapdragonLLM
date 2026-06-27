# Data pipeline

```
documents -> quality filter -> exact dedup -> tokenizer -> uint16 shards + manifest
                                                                |
                         resumable sampler <- mixture <- memmap datasets
                                   |
                       BatchBuilder -> prefetch thread -> pinned memory -> GPU
```

## Shard format

Each shard is `<stem>.bin` (little-endian `uint16` token ids, every document terminated by `<|eos|>`)
and `<stem>.idx` (`int64` document start offsets plus the total length). `manifest.json` lists the
shards with their split (`val` shards are written first, then `train`), token and document counts, and
the vocabulary size. `DatasetManifest.verify` checks that file sizes match.

## Preparing data

    python scripts/prepare_data.py --spec hf:HuggingFaceFW/fineweb-edu:sample-10BT \
        --tokenizer data/tokenizer/snapdragon-32k.json --out data/fineweb-edu

Documents shorter than 200 characters, with a low letter ratio, odd word lengths or many repeated lines
are dropped, then exact duplicates (whitespace/case-insensitive) are removed. Counts are logged.

## Determinism and resume

* A sample is addressed by a global index `g`. `Mixture` picks the source for `g` from a seeded
  per-block RNG, and each source walks a fresh random permutation of its windows every epoch.
* Batch `n` is a pure function of `(config, seed, n)`; the checkpoint only stores `n`.
* Ranks take disjoint slices of each global batch, so the union over ranks equals what a single process
  would see. Changing the GPU count mid-run does not repeat or skip data.
* `PrefetchLoader.batch_index` counts batches *delivered*, not prefetched, so resumes are exact.

## Packing

Documents are concatenated; windows of `seq_len + 1` tokens are cut from the stream, so no compute is
wasted on padding. `data/packing.py` derives document ids and per-document position ids from the
`<|eos|>` positions, and can ignore the target that crosses a document boundary.
