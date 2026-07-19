#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p artifacts/final/raw
cp results/summary_table.csv results/summary_table.md artifacts/final/
cp results/summary_aggregate.csv results/summary_aggregate.md artifacts/final/
cp results/benchmark_summary.csv results/benchmark_kv_cache_summary.csv artifacts/final/
find results -maxdepth 1 -name '*.json' -exec cp {} artifacts/final/raw/ \;
python3 -m src.plot_results --results-dir results --output-dir artifacts/final

echo "Release artifacts are ready in artifacts/final"
