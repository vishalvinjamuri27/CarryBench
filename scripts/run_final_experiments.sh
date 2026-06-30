#!/usr/bin/env bash
# Run the final Colab experiment suite and produce CSV/Markdown summaries.
#
# Defaults:
#   SEEDS="0 1 2"   paired multi-seed runs
#   RUN_LONG=1      extra 3000-step 6-digit answer-only run for seed 0
#   RUN_RUNTIME=1   standard JAX/PyTorch/KV-cache runtime benchmarks
#
# Examples:
#   ./scripts/run_final_experiments.sh
#   SEEDS="0" RUN_LONG=0 RUN_RUNTIME=0 ./scripts/run_final_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"
RUN_LONG="${RUN_LONG:-1}"
RUN_RUNTIME="${RUN_RUNTIME:-1}"
GENERATED_CONFIG_DIR="results/generated_configs"
mkdir -p "$GENERATED_CONFIG_DIR"

CONFIGS=(
  configs/colab_sweep_5digit.yaml
  configs/colab_sweep_6digit.yaml
  configs/colab_answer_only_5digit.yaml
  configs/colab_answer_only_6digit.yaml
  configs/colab_reversed_answer_5digit.yaml
  configs/colab_reversed_answer_6digit.yaml
)

make_seed_config() {
  local base_config="$1"
  local seed="$2"
  local base_name
  local out_config
  base_name="$(basename "$base_config" .yaml)"
  out_config="${GENERATED_CONFIG_DIR}/${base_name}_seed${seed}.yaml"
  python3 - "$base_config" "$out_config" "$seed" <<'PY'
import sys
import yaml

base_path, out_path, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(base_path) as f:
    cfg = yaml.safe_load(f)
cfg["seed"] = seed
with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  echo "$out_config"
}

run_pair() {
  local config="$1"
  echo "== ${config}: JAX =="
  python3 -m src.train_jax --config "$config"

  echo "== ${config}: PyTorch =="
  python3 -m src.train_torch --config "$config"
}

echo "== Final experiment suite =="
echo "Seeds: ${SEEDS}"

for seed in $SEEDS; do
  for base_config in "${CONFIGS[@]}"; do
    seeded_config="$(make_seed_config "$base_config" "$seed")"
    run_pair "$seeded_config"
  done
done

if [[ "$RUN_LONG" == "1" ]]; then
  echo "== Long 6-digit answer-only check =="
  long_config="$(make_seed_config configs/colab_answer_only_6digit_long.yaml 0)"
  run_pair "$long_config"
fi

if [[ "$RUN_RUNTIME" == "1" ]]; then
  echo "== Standard runtime benchmarks =="
  python3 -m src.benchmark --backend jax --config configs/colab_gpu.yaml
  python3 -m src.benchmark --backend torch --config configs/colab_gpu.yaml
  python3 -m src.benchmark --backend kv-cache --config configs/colab_gpu.yaml
fi

echo "== Writing result summaries =="
python3 -m src.summarize_results --results-dir results

echo "== Done =="
echo "Summary files:"
echo "  results/summary_table.csv"
echo "  results/summary_table.md"
echo "  results/summary_aggregate.csv"
echo "  results/summary_aggregate.md"
