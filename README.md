# CarryBench: Audited JAX vs PyTorch Transformer Benchmark

[![Tests](https://github.com/vishalvinjamuri27/CarryBench/actions/workflows/tests.yml/badge.svg)](https://github.com/vishalvinjamuri27/CarryBench/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalvinjamuri27/CarryBench/blob/main/colab_run.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Matched decoder-only transformers in JAX/Flax/Optax and PyTorch, benchmarked on fixed-width integer addition. The task gives deterministic data, an exact task-level metric, and interpretable failure modes (carry chains), which makes it a clean substrate for two questions:

1. How do JAX JIT, PyTorch eager, PyTorch SDPA, and `torch.compile` compare on compile cost, throughput, latency, and memory for the same architecture?
2. How do loss masking and carry-aligned answer order affect free-running algorithmic generalization?

For a fast technical read, jump to the [JAX model](src/flax_model.py), [PyTorch model](src/torch_model.py), [manual KV cache](src/kv_cache_jax.py), [framework-parity test](tests/test_framework_parity.py), and [artifact provenance](artifacts/final/README.md).

## Results

**Quality** (mean ± sample std across seeds 0, 1, 2 on the hash-partitioned, disjoint test set; free-running greedy exact match, no teacher forcing):

| Experiment (1,000 steps) | JAX test EM | PyTorch test EM |
|---|---:|---:|
| 5-digit full-sequence LM | 34.35% ± 50.10% | 1.07% ± 0.51% |
| 6-digit full-sequence LM | 2.73% ± 4.69% | 0.12% ± 0.13% |
| 5-digit answer-only | 94.95% ± 0.75% | 74.33% ± 32.51% |
| 6-digit answer-only | 91.72% ± 0.88% | 52.48% ± 47.53% |
| 5-digit reversed answer | **100.00% ± 0.00%** | **100.00% ± 0.00%** |
| 6-digit reversed answer | **100.00% ± 0.00%** | **100.00% ± 0.00%** |

Emitting the answer least-significant digit first, aligned with carry propagation, reaches 100% free-running exact match on both frameworks at both digit lengths. Normal-order answer-only training is strong in JAX but unstable in PyTorch at 1,000 steps; a single-seed 3,000-step 6-digit run reaches ~99.3% in both. Full-sequence LM loss is substantially worse. Large standard deviations are reported as-is: three seeds expose brittle optimization without pretending to estimate it precisely.

![Generated exact-match results](artifacts/final/generated_accuracy.png)

**Runtime** (synchronized measurements, 4.75M parameters, batch 256, sequence length 13):

| Backend | Precision | First step (s) | Median ms/step | p95 ms/step | Tokens/s | Peak device memory |
|---|---|---:|---:|---:|---:|---:|
| JAX JIT | FP32 | 19.08 | **7.25** | **8.11** | **451,464** | 547 MB |
| JAX JIT | BF16 | 18.72 | 12.08 | 13.22 | 272,828 | 409 MB |
| PyTorch eager, manual attention | FP32 | 0.38 | 16.51 | 18.06 | 199,229 | 450 MB |
| PyTorch eager, SDPA | FP32 | 0.38 | 12.60 | 13.76 | 260,764 | 443 MB |
| PyTorch compiled, SDPA | FP32 | 14.98 | 11.14 | 11.79 | 296,532 | 449 MB |
| PyTorch compiled, SDPA | BF16 | 16.06 | 12.04 | 13.27 | 268,388 | **307 MB** |

After warm-up, JAX FP32 delivers 1.52× the throughput of compiled PyTorch SDPA and 2.27× that of eager handwritten attention on this fixed shape, at the cost of a 19.08 s first-step compile. BF16 reduces memory but does not improve throughput at this size, consistent with the workload being too small for a blanket mixed-precision speedup claim. These are not universal framework rankings — they hold for this shape, software stack, and driver.

**KV-cache decoding.** At batch 32 and 64 generated tokens, cached decoding reaches 34,132 tokens/s versus 16,337 tokens/s for naive decoding — a 2.09× speedup that grows with batch and length. The full sweep covers batches {1, 8, 32} × lengths {5, 16, 32, 64}.

![Naive versus KV-cache decoding throughput](artifacts/final/kv_cache_throughput.png)

Compact publication artifacts (aggregate CSVs, plots, and provenance) live in [`artifacts/final/`](artifacts/final/). The Colab workflow regenerates the complete raw bundle when a full audit is needed.

## Quick Start

```bash
git clone https://github.com/vishalvinjamuri27/CarryBench.git
cd CarryBench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make test    # unit tests
make lint    # ruff check + format check
make smoke   # unit tests + JAX/PyTorch smoke train + smoke benchmarks (CPU)
```

The smoke path is CPU-runnable in under a minute; full measurements require CUDA and use the Colab workflow.

## Reproduce the GPU Study

Open [colab_run.ipynb](colab_run.ipynb), select a CUDA GPU, and run all cells. The main command is:

```bash
./scripts/run_final_experiments.sh              # 3 seeds (Colab-cost default)
SEEDS="0 1 2 3 4" ./scripts/run_final_experiments.sh   # tighter estimates
./scripts/export_release_artifacts.sh           # copy compact CSVs + regenerate plots
```

The suite runs the multi-seed quality configurations (full-sequence LM, answer-only, reversed-answer at 5 and 6 digits) plus JAX JIT, PyTorch eager, PyTorch SDPA, PyTorch compiled SDPA, and JAX naive vs KV-cached decoding sweeps. Output includes raw JSON, per-run and aggregate tables, deterministic bootstrap intervals for generated accuracy, environment metadata, and SVG/PNG plots. The Colab notebook additionally emits `results_bundle.zip` of `results/` + `artifacts/`.

## Architecture

Both implementations use token and learned positional embeddings, pre-norm causal self-attention blocks, exact (non-approximate) GELU MLPs, a final LayerNorm with matched epsilon, and an untied LM head.

```mermaid
flowchart LR
    A[Prompt tokens] --> B[Token + position embedding]
    B --> C[Pre-norm causal attention blocks]
    C --> D[Final LayerNorm]
    D --> E[LM head]
    E --> F[Greedy answer generation]
```

For `n`-digit operands, every example is exactly `n + 1` answer digits wide:

```text
<bos>007+008=0015<eos>
```

The reversed-answer ablation emits least-significant digits first so generation follows carry propagation. `532080` is the reverse of the normal `080235`:

```text
12345+67890=532080
```

## Design Notes

**Primary metric is free-running exact match.** Generated exact match greedily decodes the full answer from the prompt without exposing any ground-truth answer tokens. Teacher-forced accuracy is retained under an explicit diagnostic name only. Carry-heavy exact match stresses long carry chains as a separate held-out slice.

**Splits are hash-disjoint by construction.** A SplitMix64 hash of `pair_id ^ hash(seed)` deterministically assigns every operand pair to an 80/10/10 partition without contiguous operand bands ([data.py:146-168](src/data.py#L146-L168)). Train/eval/test share no pairs at any realistic size ([tests/test_data.py](tests/test_data.py)).

**Runtime is measured after warm-up, with the accelerator synchronized.** JAX times use `jax.block_until_ready`; PyTorch times bracket the train step with `cuda.synchronize`. Host transfers and metric materialization live outside the timed region. First-step (compile) time is reported separately from steady-state median/p95/std. Runtime results record Python, framework, platform, and Git-commit metadata.

**Frameworks receive matched inputs.** Both see the same examples, batch order, architecture shape, optimizer family (AdamW), learning rate, and step budget. Default parameter initialization differs by framework on purpose — the comparison is behavioral, not bit-exact. A parameter-transfer test asserts numerical equivalence to `rtol=2e-3` when parameters *are* copied across ([tests/test_framework_parity.py](tests/test_framework_parity.py)), which anchors correctness.

**Quality and runtime bundles are separated.** They were produced from different commits and are never treated as paired observations; the artifact README pins the source commit for each ([artifacts/final/README.md](artifacts/final/README.md)).

**KV-cache correctness is verified before it is benchmarked.** The manual cache reads directly from the trained param pytree and is asserted byte-equal to the naive full-prefix path, including under `jit` ([tests/test_generation.py](tests/test_generation.py)).

## Limitations

- Addition is compact and deterministic; results do not generalize to large corpora or open-ended generation.
- JAX JIT and PyTorch eager exercise different compilation strategies. SDPA and compiled PyTorch reduce, but do not eliminate, framework implementation differences.
- Framework default initializers differ. Paired data seeds do not make optimization trajectories numerically identical.
- Three seeds are economical for Colab but weak for unstable results; five or more are recommended for release claims.
- The model is small enough that kernel-launch overhead can dominate runtime.
- KV-cache benefits on synthetic long decoding show runtime scaling, not model quality.
- GPU results vary with accelerator, driver, CUDA/cuDNN, framework version, thermal state, and shared runtime load.
- Framework memory counters are not guaranteed to capture identical allocator semantics.

## Repository Layout

```text
artifacts/final/          Compact CSV tables, PNG plots, and provenance README
configs/                  Smoke + 5/6-digit ablation + Colab-GPU experiment YAMLs
scripts/                  Smoke, final-suite, and artifact-export runners
src/
  data.py                 Hash-disjoint splits, curriculum, carry-heavy diagnostics
  tokenizer.py            Fixed char-level vocab
  flax_model.py           JAX/Flax decoder-only transformer
  torch_model.py          PyTorch transformer with manual and SDPA attention
  train_jax.py            Jitted training + free-running eval
  train_torch.py          Eager/compiled training + free-running eval
  generate_jax.py         Naive autoregressive decoding baseline
  kv_cache_jax.py         Manual KV-cache prefill + decode step (jit-compatible)
  benchmark.py            Runtime and decode benchmark CLI
  metrics.py              Cross-entropy, token accuracy, exact match
  summarize_results.py    Per-run + aggregate tables with bootstrap CI
  plot_results.py         Accuracy and KV-cache throughput plots
  utils.py                Config validation, timing stats, environment metadata
tests/                    Data splits, tokenizer, parity, generation, SDPA, summaries
colab_run.ipynb           GPU experiment runner and audit bundler
```

## License

MIT. See [LICENSE](LICENSE).
