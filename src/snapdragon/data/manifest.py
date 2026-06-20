"""Dataset manifest: what shards exist, their sizes and splits, and which tokenizer made them."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_NAME = "manifest.json"
VERSION = 1


@dataclass
class ShardInfo:
    name: str  # stem, e.g. "train_00003"
    tokens: int
    docs: int
    split: str  # "train" | "val"


@dataclass
class DatasetManifest:
    vocab_size: int
    eos_id: int
    shards: list[ShardInfo] = field(default_factory=list)
    version: int = VERSION

    def shards_for(self, split: str) -> list[ShardInfo]:
        return [s for s in self.shards if s.split == split]

    def total_tokens(self, split: str | None = None) -> int:
        return sum(s.tokens for s in self.shards if split is None or s.split == split)

    def save(self, directory: str | Path) -> None:
        (Path(directory) / MANIFEST_NAME).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, directory: str | Path) -> DatasetManifest:
        raw = json.loads((Path(directory) / MANIFEST_NAME).read_text())
        if raw.get("version") != VERSION:
            raise ValueError(f"unsupported manifest version {raw.get('version')}")
        return cls(
            vocab_size=raw["vocab_size"],
            eos_id=raw["eos_id"],
            shards=[ShardInfo(**s) for s in raw["shards"]],
            version=raw["version"],
        )

    def verify(self, directory: str | Path) -> None:
        """Cheap integrity check: every shard file exists with the size the manifest claims."""
        d = Path(directory)
        for s in self.shards:
            bin_path, idx_path = d / f"{s.name}.bin", d / f"{s.name}.idx"
            if not bin_path.exists() or not idx_path.exists():
                raise FileNotFoundError(f"missing files for shard {s.name}")
            if bin_path.stat().st_size != s.tokens * 2:
                raise ValueError(f"shard {s.name}: .bin size does not match manifest")
            if idx_path.stat().st_size != (s.docs + 1) * 8:
                raise ValueError(f"shard {s.name}: .idx size does not match manifest")
