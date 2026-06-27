"""Resolve a dataset spec string into a stream of documents.

Specs:  ``hf:<name>[:<config>]``  stream from the HuggingFace hub
        ``files:<path>[,<path>...]``  local .txt / .jsonl files
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator

from snapdragon.tokenizer.corpus import iter_hf_texts, iter_local_texts
from snapdragon.utils.logging import get_logger

log = get_logger(__name__)


def parse_spec(spec: str) -> tuple[str, list[str]]:
    kind, sep, rest = spec.partition(":")
    if not sep or kind not in ("hf", "files") or not rest:
        raise ValueError(f"bad dataset spec {spec!r}; use hf:<name>[:<config>] or files:<a>,<b>")
    if kind == "files":
        return kind, rest.split(",")
    return kind, rest.split(":", 1)


def with_retries(make_iter: Callable[[], Iterator[str]], attempts: int = 5, backoff: float = 5.0) -> Iterator[str]:
    """Re-open a flaky stream after network errors, skipping the documents already delivered."""
    delivered = 0
    for attempt in range(attempts):
        try:
            for i, doc in enumerate(make_iter()):
                if i < delivered:
                    continue
                delivered += 1
                yield doc
            return
        except (OSError, ConnectionError) as e:
            wait = backoff * 2**attempt
            log.warning("stream failed after %d docs (%s); retrying in %.0fs", delivered, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up after {attempts} attempts at document {delivered}")


def iter_documents(spec: str) -> Iterator[str]:
    kind, args = parse_spec(spec)
    if kind == "files":
        return iter_local_texts(args)
    name = args[0]
    config = args[1] if len(args) > 1 else None
    return with_retries(lambda: iter_hf_texts(name, config))
