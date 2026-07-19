"""Unified benchmark CLI.

Usage:
    python -m src.benchmark --backend jax --config configs/colab_gpu.yaml
    python -m src.benchmark --backend torch --config configs/colab_gpu.yaml
    python -m src.benchmark --backend kv-cache --config configs/colab_gpu.yaml

`--backend jax` and `--backend torch` each train a fresh model per the given
config (reusing train_jax.train / train_torch.train) and report the
standardized training/runtime metrics below.

`--backend kv-cache` first trains a small JAX model (same as `--backend
jax`), then benchmarks naive full-prefix autoregressive decoding against
manual KV-cache decoding on addition prompts, reporting prefill/decode
latency and throughput for each.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from . import data, utils

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_jax_backend(cfg: Dict, config_name: str) -> Dict:
    from . import train_jax

    result, params, model, model_cfg = train_jax.train(cfg, config_name)
    return {
        "backend": "jax",
        "device": result["device"],
        "model_params": result["model_params"],
        "precision": result["precision"],
        "peak_device_memory_bytes": result["peak_device_memory_bytes"],
        "batch_size": result["batch_size"],
        "seq_len": result["seq_len"],
        "operand_digits": result.get("operand_digits"),
        "result_digits": result.get("result_digits"),
        "prompt_len": result.get("prompt_len"),
        "answer_order": result.get("answer_order"),
        "first_step_compile_or_warmup_time_sec": result["first_step_time_sec"],
        "steady_state_step_time_ms": result["steady_state_step_time_ms"],
        "tokens_per_sec": result["tokens_per_sec"],
        "eval_teacher_forced_exact_match_accuracy": result["eval_teacher_forced_exact_match_accuracy"],
        "eval_generated_exact_match_accuracy": result["eval_generated_exact_match_accuracy"],
        "test_generated_exact_match_accuracy": result["test_generated_exact_match_accuracy"],
        "eval_token_accuracy": result["eval_token_accuracy"],
        "carry_heavy_exact_match_accuracy": result["carry_heavy_exact_match_accuracy"],
        "carry_heavy_generated_exact_match_accuracy": result["carry_heavy_generated_exact_match_accuracy"],
        "timing_statistics": result["timing_statistics"],
        "environment": result["environment"],
    }


def run_torch_backend(cfg: Dict, config_name: str, use_compile: bool) -> Dict:
    from . import train_torch

    result, model, device = train_torch.train(cfg, config_name, use_compile=use_compile)
    return {
        "backend": "torch",
        "device": result["device"],
        "compiled": result["compiled"],
        "torch_sdpa": result["torch_sdpa"],
        "model_params": result["model_params"],
        "precision": result["precision"],
        "peak_device_memory_bytes": result["peak_device_memory_bytes"],
        "batch_size": result["batch_size"],
        "seq_len": result["seq_len"],
        "operand_digits": result.get("operand_digits"),
        "result_digits": result.get("result_digits"),
        "prompt_len": result.get("prompt_len"),
        "answer_order": result.get("answer_order"),
        "first_step_compile_or_warmup_time_sec": result["first_step_time_sec"],
        "steady_state_step_time_ms": result["steady_state_step_time_ms"],
        "tokens_per_sec": result["tokens_per_sec"],
        "eval_teacher_forced_exact_match_accuracy": result["eval_teacher_forced_exact_match_accuracy"],
        "eval_generated_exact_match_accuracy": result["eval_generated_exact_match_accuracy"],
        "test_generated_exact_match_accuracy": result["test_generated_exact_match_accuracy"],
        "eval_token_accuracy": result["eval_token_accuracy"],
        "carry_heavy_exact_match_accuracy": result["carry_heavy_exact_match_accuracy"],
        "carry_heavy_generated_exact_match_accuracy": result["carry_heavy_generated_exact_match_accuracy"],
        "timing_statistics": result["timing_statistics"],
        "environment": result["environment"],
    }


def _time_call(fn, *fn_args):
    """Call `fn(*fn_args)` and return (result, elapsed_seconds), blocking
    on any JAX arrays in the result so timing reflects actual compute."""
    import jax

    t0 = time.perf_counter()
    out = fn(*fn_args)
    out = jax.block_until_ready(out)
    elapsed = time.perf_counter() - t0
    return out, elapsed


def _time_jax_blocked(fn: Callable, *fn_args) -> Tuple[object, float]:
    import jax

    t0 = time.perf_counter()
    out = fn(*fn_args)
    out = jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def _make_naive_apply(model, params):
    import jax

    def apply_prefix(ids):
        return model.apply({"params": params}, ids, True)

    return jax.jit(apply_prefix)


def _run_naive_decode_timed(jit_apply, prompt_ids, n_generate: int) -> Tuple[float, float, float]:
    import jax.numpy as jnp

    ids = prompt_ids
    logits, prefill_time = _time_jax_blocked(jit_apply, ids)
    next_id = jnp.argmax(logits[:, -1, :], axis=-1)
    ids = jnp.concatenate([ids, next_id[:, None]], axis=1)

    decode_times = []
    for _ in range(n_generate - 1):
        logits, step_time = _time_jax_blocked(jit_apply, ids)
        next_id = jnp.argmax(logits[:, -1, :], axis=-1)
        ids = jnp.concatenate([ids, next_id[:, None]], axis=1)
        decode_times.append(step_time)

    decode_time = sum(decode_times)
    return prefill_time, decode_time, prefill_time + decode_time


def _run_kv_decode_timed(
    jit_prefill, jit_decode, params, prompt_ids, cache, n_generate: int
) -> Tuple[float, float, float]:
    import jax.numpy as jnp

    (last_logits, cache, cur_len), prefill_time = _time_jax_blocked(jit_prefill, params, prompt_ids, cache)
    next_id = jnp.argmax(last_logits, axis=-1)

    decode_times = []
    for _ in range(n_generate - 1):
        (logits, cache), step_time = _time_jax_blocked(jit_decode, params, next_id[:, None], cache, cur_len)
        next_id = jnp.argmax(logits, axis=-1)
        cur_len += 1
        decode_times.append(step_time)

    decode_time = sum(decode_times)
    return prefill_time, decode_time, prefill_time + decode_time


def run_kv_cache_backend(cfg: Dict, config_name: str) -> List[Dict]:
    """Benchmark naive vs KV-cache decoding across batch and decode lengths."""
    import jax.numpy as jnp

    from . import kv_cache_jax as kv
    from . import train_jax

    task = data.task_from_config(cfg)
    generate_lengths = sorted(
        set(int(x) for x in cfg.get("kv_generate_lengths", [task.result_digits + 1, 16, 32, 64]))
    )
    if any(length <= 1 for length in generate_lengths):
        raise ValueError("kv_generate_lengths must contain integers greater than one")
    kv_cfg = dict(cfg)
    kv_cfg["model_max_seq_len"] = task.prompt_len + max(generate_lengths)
    result, params, model, model_cfg = train_jax.train(kv_cfg, config_name)
    print(
        f"[benchmark/kv-cache] trained model with {result['model_params']:,} params, now benchmarking decoding"
    )

    batch_sizes = sorted(set([1, 8, min(32, cfg["batch_size"])]))

    rows = []
    for batch_size in batch_sizes:
        examples = data.generate_dataset(batch_size, cfg.get("seed", 0), "test", task=task)
        prompt_ids = jnp.array([ex.input_ids[: task.prompt_len] for ex in examples])
        prompt_len = prompt_ids.shape[1]

        for n_generate in generate_lengths:
            max_len = prompt_len + n_generate

            jit_apply = _make_naive_apply(model, params)
            _, _, compile_total_naive = _run_naive_decode_timed(jit_apply, prompt_ids, n_generate)
            prefill_time_naive, decode_time_naive, total_naive = _run_naive_decode_timed(
                jit_apply, prompt_ids, n_generate
            )
            avg_decode_naive_ms = 1000.0 * decode_time_naive / (n_generate - 1)
            rows.append(
                _decode_result_row(
                    "naive",
                    prompt_len,
                    batch_size,
                    n_generate,
                    compile_total_naive,
                    prefill_time_naive,
                    decode_time_naive,
                    total_naive,
                    avg_decode_naive_ms,
                )
            )

            jit_prefill = kv.make_jit_prefill(model_cfg)
            jit_decode = kv.make_jit_decode_step(model_cfg)
            compile_cache = kv.init_cache(model_cfg, batch_size, max_len)
            _, _, compile_total_kv = _run_kv_decode_timed(
                jit_prefill, jit_decode, params, prompt_ids, compile_cache, n_generate
            )
            cache = kv.init_cache(model_cfg, batch_size, max_len)
            prefill_time_kv, decode_time_kv, total_kv = _run_kv_decode_timed(
                jit_prefill, jit_decode, params, prompt_ids, cache, n_generate
            )
            avg_decode_kv_ms = 1000.0 * decode_time_kv / (n_generate - 1)
            rows.append(
                _decode_result_row(
                    "kv_cache",
                    prompt_len,
                    batch_size,
                    n_generate,
                    compile_total_kv,
                    prefill_time_kv,
                    decode_time_kv,
                    total_kv,
                    avg_decode_kv_ms,
                )
            )
            print(
                f"[benchmark/kv-cache] batch_size={batch_size} generated={n_generate}: "
                f"naive={1000.0 * total_naive:.2f}ms kv_cache={1000.0 * total_kv:.2f}ms"
            )

    return rows


def _decode_result_row(
    mode,
    prompt_len,
    batch_size,
    n_generate,
    compile_time,
    prefill_time,
    decode_time,
    total_time,
    avg_decode_ms,
):
    throughput = (batch_size * (n_generate - 1)) / decode_time if decode_time > 0 else float("nan")
    return {
        "decode_mode": mode,
        "prompt_len": prompt_len,
        "batch_size": batch_size,
        "generated_tokens": n_generate,
        "compile_plus_first_generation_latency_ms": 1000.0 * compile_time,
        "steady_prefill_latency_ms": 1000.0 * prefill_time,
        "steady_avg_decode_latency_ms": avg_decode_ms,
        "steady_total_generation_latency_ms": 1000.0 * total_time,
        "steady_decode_tokens_per_sec": throughput,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["jax", "torch", "kv-cache"])
    parser.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/smoke.yaml)")
    parser.add_argument("--compile", action="store_true", help="(torch backend only) try torch.compile")
    args = parser.parse_args()

    cfg = utils.load_config(args.config)
    config_name = Path(args.config).stem

    if args.backend == "jax":
        out = run_jax_backend(cfg, config_name)
        print(out)
        utils.save_json(out, str(RESULTS_DIR / f"benchmark_jax_{config_name}.json"))
        utils.append_csv_row(out, str(RESULTS_DIR / "benchmark_summary.csv"))
    elif args.backend == "torch":
        output_name = f"{config_name}_compile" if args.compile else config_name
        out = run_torch_backend(cfg, output_name, use_compile=args.compile)
        print(out)
        utils.save_json(out, str(RESULTS_DIR / f"benchmark_torch_{output_name}.json"))
        utils.append_csv_row(out, str(RESULTS_DIR / "benchmark_summary.csv"))
    elif args.backend == "kv-cache":
        rows = run_kv_cache_backend(cfg, config_name)
        for row in rows:
            print(row)
        utils.save_json(rows, str(RESULTS_DIR / f"benchmark_kv_cache_{config_name}.json"))
        for row in rows:
            utils.append_csv_row(row, str(RESULTS_DIR / "benchmark_kv_cache_summary.csv"))

    print(f"[benchmark] wrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
