# Limitations and Threats to Validity

- Addition is intentionally compact and deterministic; results do not establish framework performance for large language models or real corpora.
- JAX JIT and PyTorch eager exercise different compilation strategies. SDPA and compiled PyTorch baselines reduce, but do not eliminate, framework implementation differences.
- Framework default parameter initialization differs. Paired data seeds do not make optimization trajectories numerically identical.
- Three seeds are economical for Colab but weak for unstable optimization results. Five or more are recommended for release claims.
- The model and sequences are small, so kernel-launch overhead can dominate runtime.
- KV-cache benefits on synthetic long decoding demonstrate runtime scaling, not meaningful long-form model quality.
- GPU results vary with accelerator, driver, CUDA/cuDNN, framework version, thermal state, and shared-runtime load.
- Greedy exact match is appropriate for deterministic addition but is not a complete evaluation strategy for open-ended language generation.
