import importlib.util
from pathlib import Path

import pytest

from snapdragon.data.tokenize_shards import tokenize_to_shards

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inspect_data.py"
DOCS = [("abc def ghi " * (3 + i % 5)).strip() for i in range(40)]


def _cli():
    spec = importlib.util.spec_from_file_location("inspect_data_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def dataset(small_tokenizer, tmp_path):
    tokenize_to_shards(DOCS, small_tokenizer, tmp_path / "ds", shard_tokens=200, val_tokens=100)
    return tmp_path / "ds"


def test_reports_counts_and_length_percentiles(dataset):
    text = _cli().main([str(dataset)])
    assert "train:" in text
    assert "val:" in text
    assert "tokens/doc mean" in text
    assert "GB on disk" in text


def test_doc_lengths_sum_to_the_split_size(dataset):
    from snapdragon.data.manifest import DatasetManifest

    cli = _cli()
    m = DatasetManifest.load(dataset)
    for split in ("train", "val"):
        assert cli.doc_lengths(dataset, m, split).sum() == m.total_tokens(split)


def test_samples_decode_back_to_the_original_documents(dataset, small_tokenizer, tmp_path):
    small_tokenizer.save(tmp_path / "tok.json")
    text = _cli().main([str(dataset), "--tokenizer", str(tmp_path / "tok.json"), "--sample", "2"])
    assert text.count("--- doc (") == 2
    assert "abc def ghi" in text


def test_corrupt_dataset_is_reported(dataset):
    shard = next(dataset.glob("train_*.bin"))
    shard.write_bytes(shard.read_bytes()[:-2])
    with pytest.raises(ValueError, match="size"):
        _cli().main([str(dataset)])
