# Benchmark Protocol

## Quality experiments

1. Use identical generated examples and batch ordering for both frameworks.
2. Assign pairs with a stable 64-bit hash so disjoint splits remain IID-like over the operand space.
3. Train each configuration with paired seeds.
4. Select configurations using validation results only.
5. Report free-running greedy exact match as the primary metric.
6. Report teacher-forced exact match only as a diagnostic.
7. Preserve every individual seed result and report aggregate dispersion.
8. Use the disjoint test partition for the final release measurement.

The training scripts evaluate the disjoint `eval` partition throughout training and materialize the disjoint `test` metrics once after training for final reporting.

## Runtime experiments

1. Record device, software versions, platform, and Git revision.
2. Synchronize accelerator work before stopping each timer.
3. Separate first-step compile/warm-up time from steady-state measurements.
4. Exclude logging and device-to-host metric transfers from timed steps.
5. Report mean, median, p95, standard deviation, and throughput.
6. Compare JAX JIT with PyTorch eager, optimized SDPA, and compiled SDPA.
7. Keep model shape, batch size, sequence length, optimizer, and precision visible in artifacts.

## Decode experiments

1. Verify cached output against naive autoregressive output.
2. Measure prefill and iterative decode separately.
3. Sweep batch size and generated length.
4. Report compile-plus-first-generation and warmed measurements separately.

## Artifact policy

The complete local release bundle contains raw JSON and all derived formats. Git tracks the compact aggregate/per-run CSVs, PNG plots, and provenance in `artifacts/final`; checkpoints, duplicate formats, and temporary generated configs remain excluded.
