#!/usr/bin/env bash
# Fast local smoke test: trains both backends on the tiny config and runs
# the test suite. Intended to run on CPU in well under a minute.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Running unit tests =="
python3 -m unittest discover -s tests

echo "== Smoke training: JAX =="
python3 -m src.train_jax --config configs/smoke.yaml

echo "== Smoke training: PyTorch =="
python3 -m src.train_torch --config configs/smoke.yaml

echo "== Smoke benchmark: jax / torch / kv-cache =="
python3 -m src.benchmark --backend jax --config configs/smoke.yaml
python3 -m src.benchmark --backend torch --config configs/smoke.yaml
python3 -m src.benchmark --backend kv-cache --config configs/smoke.yaml

echo "== Smoke test complete =="
