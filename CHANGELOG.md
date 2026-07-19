# Changelog

## Unreleased

- Improve project navigation, engineering-scope summary, package metadata, and local artifact hygiene.

## 0.2.0 - 2026-07-19

- Make train/eval/test splits disjoint by construction.
- Use hash-based split membership to keep disjoint partitions IID-like over operand values.
- Promote free-running generated exact match to the primary quality metric.
- Add generated evaluation for JAX and PyTorch, including carry-heavy slices.
- Remove host-side PyTorch metrics from timed training steps.
- Add PyTorch SDPA and compiled runtime baselines.
- Record timing distributions, package versions, platform, and Git revision.
- Preserve partial evaluation batches and validate experiment configurations.
- Expand tests, CI, packaging, licensing, and result summarization.
- Publish corrected multi-seed quality results, runtime baselines, KV-cache sweeps, plots, and raw JSON artifacts.

## 0.1.0 - 2026-06-29

- Initial public release.
