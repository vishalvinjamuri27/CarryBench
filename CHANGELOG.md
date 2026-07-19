# Changelog

## 0.2.0 - Unreleased

- Make train/eval/test splits disjoint by construction.
- Promote free-running generated exact match to the primary quality metric.
- Add generated evaluation for JAX and PyTorch, including carry-heavy slices.
- Remove host-side PyTorch metrics from timed training steps.
- Add PyTorch SDPA and compiled runtime baselines.
- Record timing distributions, package versions, platform, and Git revision.
- Preserve partial evaluation batches and validate experiment configurations.
- Expand tests, CI, packaging, licensing, and result summarization.

## 0.1.0 - 2026-06-29

- Initial public release.
