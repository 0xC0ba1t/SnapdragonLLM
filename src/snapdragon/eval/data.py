"""Download-with-cache for benchmark files."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

DEFAULT_CACHE = Path(os.environ.get("SNAPDRAGON_CACHE", Path.home() / ".cache" / "snapdragon")) / "eval"


def fetch(url: str, cache_dir: str | Path | None = None, filename: str | None = None) -> Path:
    """Download ``url`` once; later calls return the cached file. The download goes to a temporary
    name and is renamed on success, so an interrupted download never leaves a corrupt cache entry."""
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / (filename or url.rsplit("/", 1)[-1])
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
