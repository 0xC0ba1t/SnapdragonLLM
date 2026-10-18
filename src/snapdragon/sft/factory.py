"""Build the SFT loader described by a Config."""
from __future__ import annotations

from snapdragon.config.schema import Config
from snapdragon.sft.data import read_messages, tokenize_chats
from snapdragon.sft.loader import SFTLoader
from snapdragon.sft.packing import pack_examples
from snapdragon.tokenizer.fast import FastTokenizer
from snapdragon.utils.logging import get_logger

log = get_logger(__name__)


def build_sft_loader(cfg: Config, rank: int = 0, world_size: int = 1, start_batch: int = 0) -> SFTLoader:
    tok = FastTokenizer.from_file(cfg.data.tokenizer_path)
    seq_len = cfg.train.seq_len
    examples = list(tokenize_chats(tok, read_messages(cfg.data.sft_path), max_len=seq_len + 1))
    rows = pack_examples(examples, seq_len, tok.pad_id)
    trained = int((rows.y != -100).sum())
    log.info("sft: %d examples packed into %d rows of %d (%d trained tokens)", len(examples), rows.x.shape[0],
             seq_len, trained)
    return SFTLoader(rows, cfg.train.micro_batch_size, cfg.data.seed, rank, world_size, start_batch)
