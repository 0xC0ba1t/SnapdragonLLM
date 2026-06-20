"""Random access over a split's shards as if they were one long token stream."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from snapdragon.data.manifest import DatasetManifest
from snapdragon.data.shards import TOKEN_DTYPE


class TokenDataset:
    """Memory-maps shards lazily; consecutive shards form one continuous stream.

    Opening is deferred to the first read so the object stays cheap to pickle into
    dataloader workers and no file handles are held for shards that are never touched.
    """

    def __init__(self, directory: str | Path, split: str = "train") -> None:
        self.directory = Path(directory)
        self.manifest = DatasetManifest.load(self.directory)
        self.shards = self.manifest.shards_for(split)
        if not self.shards:
            raise ValueError(f"no shards for split {split!r} in {self.directory}")
        self.split = split
        sizes = np.array([s.tokens for s in self.shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])  # start offset of each shard
        self.num_tokens = int(self.offsets[-1])
        self._maps: dict[int, np.memmap] = {}

    def _map(self, i: int) -> np.memmap:
        m = self._maps.get(i)
        if m is None:
            path = self.directory / f"{self.shards[i].name}.bin"
            m = np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
            self._maps[i] = m
        return m

    def read(self, start: int, length: int) -> np.ndarray:
        """Tokens [start, start+length) as int64 (spanning shard boundaries if needed)."""
        if start < 0 or start + length > self.num_tokens:
            raise IndexError(f"read [{start}, {start + length}) outside dataset of {self.num_tokens} tokens")
        first = int(np.searchsorted(self.offsets, start, side="right")) - 1
        out = np.empty(length, dtype=np.int64)
        filled, pos, i = 0, start, first
        while filled < length:
            local = pos - int(self.offsets[i])
            take = min(length - filled, self.shards[i].tokens - local)
            out[filled : filled + take] = self._map(i)[local : local + take]
            filled += take
            pos += take
            i += 1
        return out

    def num_windows(self, seq_len: int) -> int:
        """Non-overlapping windows of seq_len+1 tokens (inputs plus the shifted target)."""
        return max((self.num_tokens - 1) // seq_len, 0)

    def get_window(self, index: int, seq_len: int) -> np.ndarray:
        return self.read(index * seq_len, seq_len + 1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = {}  # memmaps are re-opened lazily in the receiving process
        return state
