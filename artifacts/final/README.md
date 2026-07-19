# Final Artifact Provenance

This directory contains the source data and generated presentation artifacts for the results reported in the repository README.

## Quality results

- Source: corrected post-split-fix experiment bundle.
- Git commit recorded by every run: `4f3431dca7a40fbb0e23f3a627fea2d1ebd0db8c`.
- Contents: 38 `raw/train_*.json` files, per-run and aggregate CSV/Markdown tables, and generated-accuracy plots.
- Audit: all 38 runs contain `eval_generated_exact_match_accuracy`, `test_generated_exact_match_accuracy`, and `test_generation_examples`.

## Runtime and KV-cache results

- Source: the earlier full runtime bundle at commit `c297319c5a08b35497195741ea287638192ce5be`.
- Contents: 7 `raw/benchmark_*.json` files, training-runtime and KV-cache CSV tables, and KV-cache plots.
- Rationale: commit `4f3431d` changed only dataset split membership and its tests/runner plumbing. It did not change the benchmark model, timing implementation, runtime configuration, precision variants, or KV-cache implementation, so these measurements remain valid. Accuracy fields embedded in runtime JSON are not used for quality claims.

## Interpretation

Quality and runtime were measured in separate experiment suites and must not be treated as paired observations. The README reports generated exact match only from the corrected quality bundle and performance only from the dedicated runtime bundle.
