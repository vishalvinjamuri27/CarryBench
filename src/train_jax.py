"""Train the Flax decoder-only transformer on the synthetic addition task.

Usage:
    python -m src.train_jax --config configs/smoke.yaml
    python -m src.train_jax --config configs/colab_gpu.yaml

Reports, separately:
  * first_step_time_sec  -- wall-clock time of the *first* jitted train step,
    which includes XLA compilation (this is the "JIT warmup cost").
  * steady_state_step_time_ms -- average wall-clock time per train step once
    compilation has already happened (this is the actual runtime cost).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import data
from . import metrics as metrics_lib
from . import tokenizer as tok
from . import utils
from .flax_model import DecoderOnlyTransformer, TransformerConfig

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_model_config(cfg: Dict, task: data.AdditionTask) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=tok.VOCAB_SIZE,
        max_seq_len=task.seq_len - 1,  # length of the shifted `inputs` array
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg.get("dropout", 0.0),
    )


def make_loss_weights(cfg: Dict, task: data.AdditionTask) -> jnp.ndarray:
    """Per-label-position loss weights for answer-focused training."""
    weights = np.full((task.seq_len - 1,), float(cfg.get("prompt_loss_weight", 1.0)), dtype=np.float32)
    weights[data.answer_loss_mask(task)] = float(cfg.get("answer_loss_weight", 1.0))
    weights[task.seq_len - 2] = float(cfg.get("eos_loss_weight", cfg.get("prompt_loss_weight", 1.0)))
    return jnp.array(weights)


def make_train_step(
    model: DecoderOnlyTransformer,
    optimizer: optax.GradientTransformation,
    loss_weights: jnp.ndarray,
    answer_mask: jnp.ndarray,
):
    def train_step(params, opt_state, batch, dropout_rng):
        inputs, labels = batch[:, :-1], batch[:, 1:]
        metric_mask = jnp.ones_like(labels, dtype=bool)  # no padding: every example is full-length
        loss_mask = jnp.broadcast_to(loss_weights[None, :], labels.shape)

        def loss_fn(p):
            logits = model.apply(
                {"params": p}, inputs, deterministic=False, rngs={"dropout": dropout_rng}
            )
            loss = metrics_lib.cross_entropy_loss(logits, labels, loss_mask)
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        tok_acc = metrics_lib.token_accuracy(logits, labels, metric_mask)
        em_acc = metrics_lib.exact_match_accuracy(logits, labels, answer_mask)
        out_metrics = {"loss": loss, "token_acc": tok_acc, "exact_match_acc": em_acc}
        return params, opt_state, out_metrics

    return jax.jit(train_step)


def make_eval_step(model: DecoderOnlyTransformer, answer_mask: jnp.ndarray):
    def eval_step(params, batch):
        inputs, labels = batch[:, :-1], batch[:, 1:]
        mask = jnp.ones_like(labels, dtype=bool)
        logits = model.apply({"params": params}, inputs, deterministic=True)
        loss = metrics_lib.cross_entropy_loss(logits, labels, mask)
        tok_acc = metrics_lib.token_accuracy(logits, labels, mask)
        em_acc = metrics_lib.exact_match_accuracy(logits, labels, answer_mask)
        return {"loss": loss, "token_acc": tok_acc, "exact_match_acc": em_acc}

    return jax.jit(eval_step)


def infinite_batches(examples: List[data.Example], batch_size: int, seed: int) -> Iterator[np.ndarray]:
    epoch = 0
    while True:
        yielded_any = False
        for batch in data.batch_iterator(examples, batch_size, shuffle=True, seed=seed + epoch, drop_last=True):
            yielded_any = True
            yield batch
        if not yielded_any:
            raise ValueError("Not enough examples for even one batch; reduce batch_size or grow dataset.")
        epoch += 1


def _build_curriculum_schedule(cfg: Dict, seed: int, task: data.AdditionTask) -> Tuple[int, List[Dict]]:
    curriculum = cfg.get("curriculum")
    if not curriculum or not curriculum.get("enabled", False):
        examples = data.generate_dataset(cfg["n_train"], seed, "train", task=task)
        return cfg["train_steps"], [
            {
                "name": "full_random",
                "start_step": 1,
                "end_step": cfg["train_steps"],
                "steps": cfg["train_steps"],
                "n_train": len(examples),
                "sampling": "random",
                "max_operand": task.operand_max,
                "generator": infinite_batches(examples, cfg["batch_size"], seed),
            }
        ]

    stages = curriculum.get("stages", [])
    if not stages:
        raise ValueError("curriculum.enabled=true requires at least one stage")

    schedule = []
    start_step = 1
    total_steps = 0
    for idx, stage in enumerate(stages):
        steps = int(stage["steps"])
        if steps <= 0:
            raise ValueError(f"Curriculum stage {stage.get('name', idx)!r} must have positive steps")
        name = stage.get("name", f"stage_{idx + 1}")
        n_train = int(stage.get("n_train", cfg["n_train"]))
        sampling = stage.get("sampling", "random")
        max_operand = int(stage.get("max_operand", task.operand_max))
        carry_fraction = float(stage.get("carry_fraction", 0.5))
        stage_seed = seed + 1009 * (idx + 1)
        examples = data.make_curriculum_dataset(
            n_train,
            stage_seed,
            name,
            max_operand=max_operand,
            sampling=sampling,
            carry_fraction=carry_fraction,
            task=task,
        )
        end_step = start_step + steps - 1
        schedule.append(
            {
                "name": name,
                "start_step": start_step,
                "end_step": end_step,
                "steps": steps,
                "n_train": len(examples),
                "sampling": sampling,
                "max_operand": max_operand,
                "carry_fraction": carry_fraction if sampling == "carry_mix" else None,
                "generator": infinite_batches(examples, cfg["batch_size"], stage_seed),
            }
        )
        total_steps += steps
        start_step = end_step + 1

    return total_steps, schedule


def _public_schedule(schedule: List[Dict]) -> List[Dict]:
    return [{k: v for k, v in stage.items() if k != "generator" and v is not None} for stage in schedule]


def run_eval(eval_step_fn, params, examples: List[data.Example], batch_size: int) -> Dict[str, float]:
    totals = {"loss": 0.0, "token_acc": 0.0, "exact_match_acc": 0.0}
    n_batches = 0
    for batch in data.batch_iterator(examples, batch_size, shuffle=False, drop_last=True):
        out = eval_step_fn(params, jnp.array(batch))
        for k in totals:
            totals[k] += float(out[k])
        n_batches += 1
    if n_batches == 0:
        return {k: float("nan") for k in totals}
    return {k: v / n_batches for k, v in totals.items()}


def _build_eval_slices(cfg: Dict, seed: int, task: data.AdditionTask) -> Dict[str, List[data.Example]]:
    if not cfg.get("eval_curriculum_slices", False):
        return {}
    n_eval = cfg["n_eval"]
    return {
        "one_digit": data.make_curriculum_dataset(
            n_eval, seed + 201, "eval_one_digit", max_operand=min(9, task.operand_max), sampling="random", task=task
        ),
        "two_digit": data.make_curriculum_dataset(
            n_eval, seed + 202, "eval_two_digit", max_operand=min(99, task.operand_max), sampling="random", task=task
        ),
        "no_carry": data.make_curriculum_dataset(
            n_eval, seed + 203, "eval_no_carry", max_operand=task.operand_max, sampling="no_carry", task=task
        ),
        "full_random": data.generate_dataset(n_eval, seed, "eval", task=task),
        "carry_heavy": data.make_carry_heavy_dataset(cfg["n_carry_heavy"], seed, task=task),
    }


def _run_eval_slices(eval_step_fn, params, eval_slices: Dict[str, List[data.Example]], batch_size: int) -> Dict:
    return {name: run_eval(eval_step_fn, params, examples, batch_size) for name, examples in eval_slices.items()}


def train(cfg: Dict, config_name: str):
    """Run JAX training end-to-end and return (result_dict, params, model, model_cfg).

    Factored out of `main()` so `benchmark.py` can reuse the exact same
    training/eval/timing logic instead of duplicating it.
    """
    seed = cfg.get("seed", 0)

    task = data.task_from_config(cfg)
    model_cfg = build_model_config(cfg, task)
    model = DecoderOnlyTransformer(model_cfg)

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.zeros((1, model_cfg.max_seq_len), dtype=jnp.int32)
    variables = model.init({"params": init_rng, "dropout": init_rng}, dummy_input, deterministic=True)
    params = variables["params"]
    n_params = utils.count_params(params)

    optimizer = optax.adamw(learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    opt_state = optimizer.init(params)

    loss_weights = make_loss_weights(cfg, task)
    answer_mask = jnp.array(data.answer_loss_mask(task))
    train_step_fn = make_train_step(model, optimizer, loss_weights, answer_mask)
    eval_step_fn = make_eval_step(model, answer_mask)

    print(
        f"[train_jax] config={config_name} device={jax.devices()[0].platform} "
        f"digits={task.operand_digits} params={n_params:,}"
    )

    eval_examples = data.generate_dataset(cfg["n_eval"], seed, "eval", task=task)
    carry_examples = data.make_carry_heavy_dataset(cfg["n_carry_heavy"], seed, task=task)
    eval_slices = _build_eval_slices(cfg, seed, task)

    train_steps, schedule = _build_curriculum_schedule(cfg, seed, task)
    public_schedule = _public_schedule(schedule)
    if cfg.get("curriculum", {}).get("enabled", False):
        print("[train_jax] curriculum schedule:")
        for stage in public_schedule:
            print(
                "  "
                f"{stage['start_step']:>6}-{stage['end_step']:<6} "
                f"{stage['name']} sampling={stage['sampling']} max_operand={stage['max_operand']} "
                f"n_train={stage['n_train']}"
            )

    eval_every = cfg["eval_every"]
    seq_len_inputs = model_cfg.max_seq_len

    history = []
    step_times = []
    first_step_time = None
    stage_idx = 0
    current_stage = schedule[stage_idx]

    for step in range(1, train_steps + 1):
        while step > current_stage["end_step"]:
            stage_idx += 1
            current_stage = schedule[stage_idx]
            print(f"[train_jax] entering curriculum stage {current_stage['name']!r} at step {step}")

        batch = jnp.array(next(current_stage["generator"]))
        rng, dropout_rng = jax.random.split(rng)

        t0 = time.perf_counter()
        params, opt_state, step_metrics = train_step_fn(params, opt_state, batch, dropout_rng)
        step_metrics = jax.block_until_ready(step_metrics)  # force completion for accurate timing
        step_time = time.perf_counter() - t0

        if step == 1:
            first_step_time = step_time
        else:
            step_times.append(step_time)

        if step % eval_every == 0 or step == train_steps:
            eval_metrics = run_eval(eval_step_fn, params, eval_examples, cfg["batch_size"])
            carry_metrics = run_eval(eval_step_fn, params, carry_examples, cfg["batch_size"])
            slice_metrics = _run_eval_slices(eval_step_fn, params, eval_slices, cfg["batch_size"])
            print(
                f"step {step:5d} [{current_stage['name']}] | train_loss={float(step_metrics['loss']):.4f} "
                f"train_exact={float(step_metrics['exact_match_acc']):.3f} | "
                f"eval_loss={eval_metrics['loss']:.4f} eval_exact={eval_metrics['exact_match_acc']:.3f} | "
                f"carry_exact={carry_metrics['exact_match_acc']:.3f}"
            )
            history.append(
                {
                    "step": step,
                    "stage": current_stage["name"],
                    "train_loss": float(step_metrics["loss"]),
                    "train_token_acc": float(step_metrics["token_acc"]),
                    "train_exact_match_acc": float(step_metrics["exact_match_acc"]),
                    "eval_loss": eval_metrics["loss"],
                    "eval_token_acc": eval_metrics["token_acc"],
                    "eval_exact_match_acc": eval_metrics["exact_match_acc"],
                    "carry_heavy_exact_match_acc": carry_metrics["exact_match_acc"],
                    "eval_slices": slice_metrics,
                    "step_time_sec": step_time,
                }
            )

    warmup = min(5, len(step_times))
    steady_state = step_times[warmup:] if len(step_times) > warmup else step_times
    steady_state_step_time_ms = 1000.0 * (sum(steady_state) / len(steady_state)) if steady_state else float("nan")
    tokens_per_sec = (
        cfg["batch_size"] * seq_len_inputs / (steady_state_step_time_ms / 1000.0)
        if steady_state_step_time_ms == steady_state_step_time_ms
        else float("nan")
    )

    final_eval = run_eval(eval_step_fn, params, eval_examples, cfg["batch_size"])
    final_carry = run_eval(eval_step_fn, params, carry_examples, cfg["batch_size"])
    final_eval_slices = _run_eval_slices(eval_step_fn, params, eval_slices, cfg["batch_size"])

    result = {
        "backend": "jax",
        "config": config_name,
        "seed": seed,
        "device": jax.devices()[0].platform,
        "model_params": n_params,
        "batch_size": cfg["batch_size"],
        "seq_len": seq_len_inputs,
        "operand_digits": task.operand_digits,
        "result_digits": task.result_digits,
        "prompt_len": task.prompt_len,
        "answer_order": task.answer_order,
        "train_steps": train_steps,
        "curriculum_enabled": bool(cfg.get("curriculum", {}).get("enabled", False)),
        "curriculum_schedule": public_schedule,
        "loss_weights": {
            "prompt_loss_weight": float(cfg.get("prompt_loss_weight", 1.0)),
            "answer_loss_weight": float(cfg.get("answer_loss_weight", 1.0)),
            "eos_loss_weight": float(cfg.get("eos_loss_weight", cfg.get("prompt_loss_weight", 1.0))),
        },
        "first_step_time_sec": first_step_time,
        "steady_state_step_time_ms": steady_state_step_time_ms,
        "tokens_per_sec": tokens_per_sec,
        "eval_loss": final_eval["loss"],
        "eval_token_accuracy": final_eval["token_acc"],
        "eval_exact_match_accuracy": final_eval["exact_match_acc"],
        "carry_heavy_exact_match_accuracy": final_carry["exact_match_acc"],
        "eval_slices": final_eval_slices,
        "history": history,
    }

    return result, params, model, model_cfg


def save_outputs(result: Dict, params, n_params: int, config_name: str) -> None:
    out_path = RESULTS_DIR / f"train_jax_{config_name}.json"
    utils.save_json(result, str(out_path))
    print(f"[train_jax] wrote results to {out_path}")

    # Save a lightweight checkpoint only if it's small enough not to bloat the repo.
    if n_params < 1_000_000:
        import flax.serialization as serialization

        ckpt_path = RESULTS_DIR / f"checkpoint_jax_{config_name}.msgpack"
        with open(ckpt_path, "wb") as f:
            f.write(serialization.to_bytes(params))
        print(f"[train_jax] wrote small checkpoint to {ckpt_path}")
    else:
        print(f"[train_jax] skipping checkpoint save ({n_params:,} params, too large for repo)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/smoke.yaml)")
    args = parser.parse_args()

    cfg = utils.load_config(args.config)
    config_name = Path(args.config).stem

    result, params, model, model_cfg = train(cfg, config_name)
    save_outputs(result, params, result["model_params"], config_name)


if __name__ == "__main__":
    main()
