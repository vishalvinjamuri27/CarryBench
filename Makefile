PYTHON ?= python3

.PHONY: help test lint format smoke bench

help:
	@echo "make test    - run the unit test suite"
	@echo "make lint    - ruff check + ruff format --check"
	@echo "make format  - ruff format (rewrites files)"
	@echo "make smoke   - CPU end-to-end smoke: tests, JAX/PyTorch train, benchmarks"
	@echo "make bench   - full GPU experiment suite (requires CUDA)"

test:
	$(PYTHON) -m unittest discover -s tests

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests

smoke:
	./scripts/run_smoke.sh

bench:
	./scripts/run_final_experiments.sh
