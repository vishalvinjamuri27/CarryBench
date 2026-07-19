# Final Artifact Provenance

This directory contains the compact publication artifacts for the results reported in the repository README. Full reruns produce raw JSON, Markdown, SVG, environment, and checkpoint files locally; those reproducible outputs are intentionally not duplicated in Git.

## Quality results

- Source: corrected post-split-fix experiment bundle.
- Git commit recorded by every run: `4f3431dca7a40fbb0e23f3a627fea2d1ebd0db8c`.
- Published contents: per-run and aggregate CSV tables plus the generated-accuracy PNG.
- Source audit: all 38 bundle runs contained `eval_generated_exact_match_accuracy`, `test_generated_exact_match_accuracy`, and `test_generation_examples`.

## Runtime and KV-cache results

- Source: the earlier full runtime bundle at commit `c297319c5a08b35497195741ea287638192ce5be`.
- Published contents: training-runtime and KV-cache CSV tables plus the KV-cache PNG.
- Rationale: commit `4f3431d` changed only dataset split membership and its tests/runner plumbing. It did not change the benchmark model, timing implementation, runtime configuration, precision variants, or KV-cache implementation, so these measurements remain valid. Accuracy fields embedded in runtime JSON are not used for quality claims.

## Interpretation

Quality and runtime were measured in separate experiment suites and must not be treated as paired observations. The README reports generated exact match only from the corrected quality bundle and performance only from the dedicated runtime bundle. Run `./scripts/export_release_artifacts.sh` after a full experiment to regenerate the expanded local artifact set.
