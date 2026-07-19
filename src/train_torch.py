"""Train the PyTorch baseline transformer on the same synthetic addition task.

Usage:
    python -m src.train_torch --config configs/smoke.yaml
    python -m src.train_torch --config configs/colab_gpu.yaml [--compile]

This is the comparison baseline, not the main focus of the project -- see
README for the JAX vs PyTorch discussion. `--compile` optionally wraps the
model in `torch.compile`; if compilation is unavailable or fails for any
reason, we fall back to eager mode and keep going (never hard-fails).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import data, utils
from . import tokenizer as tok
from .torch_model import DecoderOnlyTransformer, count_params

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _autocast(model, device: torch.device):
    precision = getattr(model, "benchmark_precision", "float32")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError(f"Unsupported precision {precision!r}")
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bfloat16",
    )


def infinite_batches(examples: List[data.Example], batch_size: int, seed: int) -> Iterator[np.ndarray]:
    epoch = 0
    while True:
        yielded_any = False
        for batch in data.batch_iterator(
            examples, batch_size, shuffle=True, seed=seed + epoch, drop_last=True
        ):
            yielded_any = True
            yield batch
        if not yielded_any:
            raise ValueError("Not enough examples for even one batch; reduce batch_size or grow dataset.")
        epoch += 1


def make_loss_weights(cfg: Dict, device: torch.device, task: data.AdditionTask) -> torch.Tensor:
    weights = np.full((task.seq_len - 1,), float(cfg.get("prompt_loss_weight", 1.0)), dtype=np.float32)
    weights[data.answer_loss_mask(task)] = float(cfg.get("answer_loss_weight", 1.0))
    weights[task.seq_len - 2] = float(cfg.get("eos_loss_weight", cfg.get("prompt_loss_weight", 1.0)))
    return torch.from_numpy(weights).to(device)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_mask: np.ndarray,
    loss_weights: torch.Tensor | None = None,
) -> Dict:
    if loss_weights is None:
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    else:
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        weights = loss_weights[None, :].expand_as(per_token_loss)
        loss = torch.sum(per_token_loss * weights) / torch.clamp(torch.sum(weights), min=1.0)
    preds = torch.argmax(logits, dim=-1)
    token_acc = (preds == labels).float().mean().item()

    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    exact_match = float(np.mean(np.all(preds_np[:, answer_mask] == labels_np[:, answer_mask], axis=1)))
    return {"loss": loss, "token_acc": token_acc, "exact_match_acc": exact_match}


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


def _build_eval_slices(cfg: Dict, seed: int, task: data.AdditionTask) -> Dict[str, List[data.Example]]:
    if not cfg.get("eval_curriculum_slices", False):
        return {}
    n_eval = cfg["n_eval"]
    return {
        "one_digit": data.make_curriculum_dataset(
            n_eval,
            seed + 201,
            "eval_one_digit",
            max_operand=min(9, task.operand_max),
            sampling="random",
            task=task,
        ),
        "two_digit": data.make_curriculum_dataset(
            n_eval,
            seed + 202,
            "eval_two_digit",
            max_operand=min(99, task.operand_max),
            sampling="random",
            task=task,
        ),
        "no_carry": data.make_curriculum_dataset(
            n_eval, seed + 203, "eval_no_carry", max_operand=task.operand_max, sampling="no_carry", task=task
        ),
        "full_random": data.generate_dataset(n_eval, seed, "eval", task=task),
        "carry_heavy": data.make_carry_heavy_dataset(cfg["n_carry_heavy"], seed, task=task),
    }


def _run_eval_slices(
    model,
    eval_slices: Dict[str, List[data.Example]],
    batch_size: int,
    device: torch.device,
    answer_mask: np.ndarray,
) -> Dict:
    return {
        name: run_eval(model, examples, batch_size, device, answer_mask)
        for name, examples in eval_slices.items()
    }


@torch.no_grad()
def run_eval(
    model, examples: List[data.Example], batch_size: int, device: torch.device, answer_mask: np.ndarray
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "token_acc": 0.0, "exact_match_acc": 0.0}
    n_examples = 0
    with torch.no_grad():
        for batch in data.batch_iterator(examples, batch_size, shuffle=False, drop_last=False):
            batch_t = torch.from_numpy(batch).long().to(device)
            inputs, labels = batch_t[:, :-1], batch_t[:, 1:]
            with _autocast(model, device):
                logits = model(inputs)
            m = compute_metrics(logits, labels, answer_mask)
            for key in totals:
                totals[key] += (float(m["loss"]) if key == "loss" else m[key]) * len(batch)
            n_examples += len(batch)
    model.train()
    if n_examples == 0:
        return {k: float("nan") for k in totals}
    return {k: v / n_examples for k, v in totals.items()}


def run_generation_eval(
    model, examples, batch_size: int, device: torch.device, task: data.AdditionTask
) -> float:
    """Greedily generate answers without teacher forcing and return exact match."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data.batch_iterator(examples, batch_size, shuffle=False, drop_last=False):
            batch_t = torch.from_numpy(batch).long().to(device)
            ids = batch_t[:, : task.prompt_len]
            for _ in range(task.result_digits):
                with _autocast(model, device):
                    logits = model(ids)
                ids = torch.cat([ids, torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)], dim=1)
            pred = ids[:, task.prompt_len : task.answer_end]
            target = batch_t[:, task.answer_start : task.answer_end]
            correct += int(torch.sum(torch.all(pred == target, dim=1)).item())
            total += len(batch)
    model.train()
    return correct / total if total else float("nan")


def generation_examples(model, examples, device: torch.device, task: data.AdditionTask, limit: int = 10):
    selected = examples[:limit]
    batch = torch.from_numpy(data.examples_to_array(selected)).long().to(device)
    ids = batch[:, : task.prompt_len]
    model.eval()
    with torch.no_grad():
        for _ in range(task.result_digits):
            with _autocast(model, device):
                logits = model(ids)
            ids = torch.cat([ids, torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)], dim=1)
    predictions = ids[:, task.prompt_len : task.answer_end].cpu().numpy()
    model.train()
    rows = []
    for example, prediction in zip(selected, predictions, strict=True):
        predicted_text = tok.decode(prediction.tolist())
        rows.append(
            {
                "prompt": example.prompt,
                "target": example.target_answer,
                "prediction": predicted_text,
                "correct": predicted_text == example.target_answer,
            }
        )
    return rows


def train(cfg: Dict, config_name: str, use_compile: bool = False):
    """Run PyTorch training end-to-end and return (result_dict, model, device).

    Factored out of `main()` so `benchmark.py` can reuse the exact same
    training/eval/timing logic instead of duplicating it.
    """
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    task = data.task_from_config(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = task.seq_len - 1

    model = DecoderOnlyTransformer(
        vocab_size=tok.VOCAB_SIZE,
        max_seq_len=max_seq_len,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg.get("dropout", 0.0),
        use_sdpa=bool(cfg.get("torch_sdpa", False)),
    ).to(device)
    model.benchmark_precision = cfg.get("precision", "float32")
    n_params = count_params(model)

    compiled = False
    if use_compile:
        try:
            model = torch.compile(model)
            compiled = True
        except Exception as e:  # torch.compile can fail for many environment-specific reasons
            print(f"[train_torch] torch.compile unavailable/failed ({e}); falling back to eager mode")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss_weights = make_loss_weights(cfg, device, task)
    answer_mask = data.answer_loss_mask(task)

    print(
        f"[train_torch] config={config_name} device={device} digits={task.operand_digits} "
        f"params={n_params:,} compiled={compiled}"
    )

    eval_examples = data.generate_dataset(cfg["n_eval"], seed, "eval", task=task)
    test_examples = data.generate_dataset(cfg.get("n_test", cfg["n_eval"]), seed, "test", task=task)
    carry_examples = data.make_carry_heavy_dataset(cfg["n_carry_heavy"], seed, task=task)
    eval_slices = _build_eval_slices(cfg, seed, task)

    train_steps, schedule = _build_curriculum_schedule(cfg, seed, task)
    public_schedule = _public_schedule(schedule)
    if cfg.get("curriculum", {}).get("enabled", False):
        print("[train_torch] curriculum schedule:")
        for stage in public_schedule:
            print(
                "  "
                f"{stage['start_step']:>6}-{stage['end_step']:<6} "
                f"{stage['name']} sampling={stage['sampling']} max_operand={stage['max_operand']} "
                f"n_train={stage['n_train']}"
            )
    eval_every = cfg["eval_every"]

    model.train()
    history = []
    step_times = []
    first_step_time = None
    stage_idx = 0
    current_stage = schedule[stage_idx]

    for step in range(1, train_steps + 1):
        while step > current_stage["end_step"]:
            stage_idx += 1
            current_stage = schedule[stage_idx]
            print(f"[train_torch] entering curriculum stage {current_stage['name']!r} at step {step}")

        batch = next(current_stage["generator"])
        batch_t = torch.from_numpy(batch).long().to(device)
        inputs, labels = batch_t[:, :-1], batch_t[:, 1:]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        with _autocast(model, device):
            logits = model(inputs)
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
            ).reshape_as(labels)
            weights = loss_weights[None, :].expand_as(per_token_loss)
            loss = torch.sum(per_token_loss * weights) / torch.clamp(torch.sum(weights), min=1.0)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - t0

        # Host transfers and reporting metrics are intentionally outside the
        # timed region so the benchmark measures the train step itself.
        m = compute_metrics(logits.detach(), labels, answer_mask, loss_weights=loss_weights)
        m["loss"] = loss.detach()

        if step == 1:
            first_step_time = step_time
        else:
            step_times.append(step_time)

        if step % eval_every == 0 or step == train_steps:
            eval_metrics = run_eval(model, eval_examples, cfg["batch_size"], device, answer_mask)
            generated_exact = run_generation_eval(model, eval_examples, cfg["batch_size"], device, task)
            carry_generated_exact = run_generation_eval(
                model, carry_examples, cfg["batch_size"], device, task
            )
            carry_metrics = run_eval(model, carry_examples, cfg["batch_size"], device, answer_mask)
            slice_metrics = _run_eval_slices(model, eval_slices, cfg["batch_size"], device, answer_mask)
            print(
                f"step {step:5d} [{current_stage['name']}] | train_loss={float(m['loss']):.4f} "
                f"train_exact={m['exact_match_acc']:.3f} | "
                f"eval_loss={eval_metrics['loss']:.4f} generated_exact={generated_exact:.3f} | "
                f"carry_exact={carry_metrics['exact_match_acc']:.3f}"
            )
            history.append(
                {
                    "step": step,
                    "stage": current_stage["name"],
                    "train_loss": float(m["loss"]),
                    "train_token_acc": m["token_acc"],
                    "train_exact_match_acc": m["exact_match_acc"],
                    "eval_loss": eval_metrics["loss"],
                    "eval_token_acc": eval_metrics["token_acc"],
                    "eval_exact_match_acc": eval_metrics["exact_match_acc"],
                    "eval_generated_exact_match_acc": generated_exact,
                    "carry_heavy_exact_match_acc": carry_metrics["exact_match_acc"],
                    "carry_heavy_generated_exact_match_acc": carry_generated_exact,
                    "eval_slices": slice_metrics,
                    "step_time_sec": step_time,
                }
            )

    warmup = min(5, len(step_times))
    steady_state = step_times[warmup:] if len(step_times) > warmup else step_times
    timing_stats = utils.timing_statistics(steady_state)
    steady_state_step_time_ms = timing_stats["mean_ms"]
    tokens_per_sec = (
        cfg["batch_size"] * max_seq_len / (steady_state_step_time_ms / 1000.0)
        if steady_state_step_time_ms == steady_state_step_time_ms
        else float("nan")
    )

    final_eval = run_eval(model, eval_examples, cfg["batch_size"], device, answer_mask)
    final_generated_exact = run_generation_eval(model, eval_examples, cfg["batch_size"], device, task)
    final_carry_generated_exact = run_generation_eval(model, carry_examples, cfg["batch_size"], device, task)
    final_test = run_eval(model, test_examples, cfg["batch_size"], device, answer_mask)
    final_test_generated_exact = run_generation_eval(model, test_examples, cfg["batch_size"], device, task)
    final_carry = run_eval(model, carry_examples, cfg["batch_size"], device, answer_mask)
    final_eval_slices = _run_eval_slices(model, eval_slices, cfg["batch_size"], device, answer_mask)

    result = {
        "backend": "torch",
        "config": config_name,
        "seed": seed,
        "device": str(device),
        "compiled": compiled,
        "torch_sdpa": bool(cfg.get("torch_sdpa", False)),
        "precision": cfg.get("precision", "float32"),
        "model_params": n_params,
        "batch_size": cfg["batch_size"],
        "seq_len": max_seq_len,
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
        "timing_statistics": timing_stats,
        "tokens_per_sec": tokens_per_sec,
        "eval_loss": final_eval["loss"],
        "eval_token_accuracy": final_eval["token_acc"],
        "eval_teacher_forced_exact_match_accuracy": final_eval["exact_match_acc"],
        "eval_generated_exact_match_accuracy": final_generated_exact,
        "test_teacher_forced_exact_match_accuracy": final_test["exact_match_acc"],
        "test_generated_exact_match_accuracy": final_test_generated_exact,
        "test_generation_examples": generation_examples(model, test_examples, device, task),
        "carry_heavy_exact_match_accuracy": final_carry["exact_match_acc"],
        "carry_heavy_generated_exact_match_accuracy": final_carry_generated_exact,
        "environment": utils.environment_metadata(),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "eval_slices": final_eval_slices,
        "history": history,
    }

    return result, model, device


def save_outputs(result: Dict, model, device: torch.device, n_params: int, config_name: str) -> None:
    out_path = RESULTS_DIR / f"train_torch_{config_name}.json"
    utils.save_json(result, str(out_path))
    print(f"[train_torch] wrote results to {out_path}")

    if n_params < 1_000_000:
        ckpt_path = RESULTS_DIR / f"checkpoint_torch_{config_name}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[train_torch] wrote small checkpoint to {ckpt_path}")
    else:
        print(f"[train_torch] skipping checkpoint save ({n_params:,} params, too large for repo)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/smoke.yaml)")
    parser.add_argument(
        "--compile", action="store_true", help="Try torch.compile (falls back to eager on failure)"
    )
    args = parser.parse_args()

    cfg = utils.load_config(args.config)
    config_name = Path(args.config).stem

    result, model, device = train(cfg, config_name, use_compile=args.compile)
    save_outputs(result, model, device, result["model_params"], config_name)


if __name__ == "__main__":
    main()
