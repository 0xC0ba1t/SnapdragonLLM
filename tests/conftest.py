import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.config.schema import Config
from snapdragon.data.loader import Batch
from snapdragon.model.factory import build_model
from snapdragon.optim.factory import build_optimizer
from snapdragon.optim.schedulers import LRSchedule
from snapdragon.train.trainer import Trainer


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

class CycleLoader:
    """Batches cut from a deterministic sequence where the next token is a fixed function of the
    previous one (a permutation cycle), so a tiny model learns it within a few dozen steps."""

    def __init__(self, batch_size, seq_len, vocab, seed=0):
        step = 5 if vocab % 5 else 3  # coprime with the vocab, so the cycle covers every token
        self.pattern = (torch.arange(4096) * step + 3 + seed) % vocab
        self.batch_size, self.seq_len = batch_size, seq_len
        self.batch_index = 0

    def __next__(self):
        B, T = self.batch_size, self.seq_len
        rows = []
        for i in range(B):
            start = ((self.batch_index * B + i) * 7) % 1000
            rows.append(self.pattern[start : start + T + 1])
        self.batch_index += 1
        w = torch.stack(rows)
        return Batch(w[:, :-1].contiguous(), w[:, 1:].contiguous())

    def state_dict(self):
        return {"batch_index": self.batch_index}

    def load_state_dict(self, state):
        self.batch_index = state["batch_index"]


@pytest.fixture
def tiny_config():
    def make(**train_overrides) -> Config:
        cfg = Config()
        cfg.model = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                                mlp_hidden=64, max_seq_len=16)
        cfg.train.seq_len = cfg.data.seq_len = 16
        cfg.train.micro_batch_size = 2
        cfg.train.global_batch_tokens = 2 * 16 * 2  # 2 micro-batches per optimizer step
        cfg.train.max_tokens = cfg.train.global_batch_tokens * 30  # 30 steps
        cfg.train.precision = "fp32"
        cfg.train.z_loss_coeff = 0.0
        cfg.optim.scheduler.warmup_steps = 2
        cfg.optim.adam_lr = 3e-3
        for k, v in train_overrides.items():
            setattr(cfg.train, k, v)
        cfg.validate()
        return cfg

    return make


@pytest.fixture
def make_trainer(tiny_config):
    def make(cfg=None, callbacks=None, seed=0):
        cfg = cfg or tiny_config()
        torch.manual_seed(seed)
        model = build_model(cfg)
        opt = build_optimizer(model, cfg.optim)
        sched = LRSchedule(opt, cfg.optim.scheduler, cfg.train.total_steps)
        loader = CycleLoader(cfg.train.micro_batch_size, cfg.train.seq_len, cfg.model.vocab_size, seed)
        return Trainer(cfg, model, opt, sched, loader, "cpu", callbacks)

    return make
