"""Deterministic weighted choice of which dataset each sample comes from."""
from __future__ import annotations

import numpy as np

BLOCK = 4096  # choices are generated in blocks so any position can be reproduced cheaply


class Mixture:
    """Maps a global sample index to a source index, reproducibly, without storing history.

    Choices for block ``b`` come from ``default_rng([seed, b])``, so resuming at sample ``g`` only
    needs to regenerate the blocks up to ``g`` (microseconds) rather than replaying training.
    """

    def __init__(self, weights: list[float], seed: int) -> None:
        if not weights or min(weights) <= 0:
            raise ValueError("weights must be a non-empty list of positive numbers")
        w = np.asarray(weights, dtype=np.float64)
        self.p = w / w.sum()
        self.seed = seed

    def block(self, b: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, b])
        return rng.choice(len(self.p), size=BLOCK, p=self.p).astype(np.int32)

    def source_of(self, g: int) -> int:
        return int(self.block(g // BLOCK)[g % BLOCK])

    def counts_before(self, g: int) -> np.ndarray:
        """How many of samples [0, g) each source received (its local cursor at g)."""
        counts = np.zeros(len(self.p), dtype=np.int64)
        for b in range(g // BLOCK):
            counts += np.bincount(self.block(b), minlength=len(self.p))
        tail = g % BLOCK
        if tail:
            counts += np.bincount(self.block(g // BLOCK)[:tail], minlength=len(self.p))
        return counts
