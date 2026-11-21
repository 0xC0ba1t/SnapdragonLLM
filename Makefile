.PHONY: install lint test slow gpu smoke docs clean

install:
	pip install -e ".[data,dev]"

lint:
	ruff check .

test: lint
	pytest -q -m "not gpu and not slow"

slow:
	pytest -q -m slow

gpu:
	pytest -q -m gpu

smoke:
	pytest -q tests/test_smoke_e2e.py -m slow

docs:
	python scripts/gen_config_docs.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
