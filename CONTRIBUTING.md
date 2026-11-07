# Contributing

## Setup

    pip install -e ".[data,dev]"
    pre-commit install            # ruff + whitespace hooks

## The checks that must pass

    ruff check .
    pytest -q -m "not gpu and not slow"      # fast CPU suite
    pytest -q -m slow                         # end-to-end smoke run (about a minute)
    pytest -q -m gpu                          # only on a CUDA machine

CI runs the first two on Linux for every push.

## Conventions

* **Every change keeps lint and tests green and ships with a test.** Numerical code is tested against a
  slow, obviously-correct reference (the fused loss against `F.cross_entropy`, the incremental BPE trainer
  against the naive one, cached decoding against full recompute).
* **Pinned contracts:** the `state_dict` key layout (`tests/model/test_state_dict_keys.py`) and the
  generated config reference (`python scripts/gen_config_docs.py` after changing a config dataclass).
  Changing either is a breaking change: say so in the changelog and provide a migration.
* Small, single-purpose commits with short imperative messages (`add rmsnorm layer`, `fix off-by-one in
  packed doc boundaries`).
* Do not claim a measured number you did not measure. Memory and time figures in the docs say whether they
  are estimates.
* No code is copied from other projects; cite the idea in a comment instead.

## Layout

See the package map in [docs/architecture.md](docs/architecture.md). New eval tasks register themselves in
`snapdragon.eval.tasks.TASKS`; new training behaviour should be a `Callback`, not an `if` in the trainer.
