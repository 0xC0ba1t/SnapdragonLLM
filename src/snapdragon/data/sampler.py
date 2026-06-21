"""Deterministic, resumable, rank-aware sample selection.

Every sample is addressed by a *global sample index* ``g`` (0, 1, 2, ...). ``locate(g)`` says which
source and which window it is. The only state needed to resume is the number of global samples
consumed, and it does not depend on how many ranks were used.
"""
from __future__ import annotations

import numpy as np

from snapdragon.data.mixture import BLOCK, Mixture


class EpochPermutation:
    """Visits n items in a fresh random order every epoch (reproducible from seed+epoch)."""

    def __init__(self, n: int, seed: int, stream: int) -> None:
        if n < 1:
            raise ValueError("a source needs at least one window")
        self.n, self.seed, self.stream = n, seed, stream
        self._epoch = -1
        self._perm = np.empty(0, dtype=np.int64)

    def index(self, cursor: int) -> int:
        epoch, pos = divmod(cursor, self.n)
        if epoch != self._epoch:
            rng = np.random.default_rng([self.seed, self.stream, epoch])
            self._perm = rng.permutation(self.n)
            self._epoch = epoch
        return int(self._perm[pos])


class Sampler:
    def __init__(self, num_windows: list[int], weights: list[float], seed: int) -> None:
        if len(num_windows) != len(weights):
            raise ValueError("num_windows and weights must have the same length")
        self.mixture = Mixture(weights, seed)
        self.perms = [EpochPermutation(n, seed, i) for i, n in enumerate(num_windows)]
        self._g = 0
        self._counts = np.zeros(len(weights), dtype=np.int64)

    def seek(self, g: int) -> None:
        """Position the internal cursor so that sample ``g`` is next (cheap for small jumps)."""
        if g < self._g:
            self._g, self._counts = 0, np.zeros_like(self._counts)
            if g > 0:
                self._counts = self.mixture.counts_before(g)
                self._g = g
        while self._g < g:
            b, off = divmod(self._g, BLOCK)
            take = min(g - self._g, BLOCK - off)
            self._counts += np.bincount(self.mixture.block(b)[off : off + take], minlength=len(self._counts))
            self._g += take

    def locate(self, g: int) -> tuple[int, int]:
        """(source index, window index within that source) for global sample ``g``."""
        self.seek(g)
        src = self.mixture.source_of(g)
        window = self.perms[src].index(int(self._counts[src]))
        return src, window


def batch_sample_indices(batch: int, micro_batch_size: int, rank: int, world_size: int) -> range:
    """Global sample indices rank ``rank`` reads for global micro-batch number ``batch``.

    Batch ``n`` covers the contiguous samples [n*W*B, (n+1)*W*B) and each rank takes its own slice
    of B, so the union over ranks is the same stream a single process would see.
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} out of range for world size {world_size}")
    start = (batch * world_size + rank) * micro_batch_size
    return range(start, start + micro_batch_size)
