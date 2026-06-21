import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def small_tokenizer():
    """A tiny trained FastTokenizer (vocab 256 + 7 + 60) shared by data/eval tests."""
    import random

    from snapdragon.tokenizer.fast import FastTokenizer, train_fast_tokenizer

    rng = random.Random(0)
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 7))) for _ in range(300)]
    corpus = [" ".join(rng.choice(words) for _ in range(12)) for _ in range(400)]
    return FastTokenizer(train_fast_tokenizer(corpus, 256 + 7 + 60))


@pytest.fixture
def make_token_dataset():
    """Factory: write one shard holding tokens offset..offset+n-2 plus eos, return a TokenDataset."""
    from snapdragon.data.manifest import DatasetManifest, ShardInfo
    from snapdragon.data.memmap_dataset import TokenDataset
    from snapdragon.data.shards import ShardWriter

    def make(path, n_tokens, offset=0):
        with ShardWriter(path, "train", shard_tokens=10**6, eos_id=65000) as w:
            w.add_document(list(range(offset, offset + n_tokens - 1)))
        m = DatasetManifest(vocab_size=65001, eos_id=65000)
        m.shards = [ShardInfo(s, t, d, "train") for s, t, d in w.files]
        m.save(path)
        return TokenDataset(path)

    return make
