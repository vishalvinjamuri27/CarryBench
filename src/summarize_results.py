"""Summarize generated benchmark JSON files into CSV and Markdown tables.

Usage:
    python -m src.summarize_results --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


TRAIN_FIELDS = [
    "path",
    "backend",
    "config",
    "config_family",
    "run_group",
    "seed",
    "device",
    "model_params",
    "batch_size",
    "seq_len",
    "operand_digits",
    "result_digits",
    "answer_order",
    "loss_type",
    "train_steps",
    "first_step_time_sec",
    "steady_state_step_time_ms",
    "tokens_per_sec",
    "eval_exact_match_accuracy",
    "carry_heavy_exact_match_accuracy",
    "time_to_50",
    "time_to_90",
    "time_to_99",
]

AGG_FIELDS = [
    "backend",
    "config_family",
    "run_group",
    "n_runs",
    "seeds",
    "operand_digits",
    "answer_order",
    "loss_type",
    "train_steps",
    "eval_exact_match_accuracy_mean",
    "eval_exact_match_accuracy_std",
    "carry_heavy_exact_match_accuracy_mean",
    "carry_heavy_exact_match_accuracy_std",
    "steady_state_step_time_ms_mean",
    "steady_state_step_time_ms_std",
    "tokens_per_sec_mean",
    "tokens_per_sec_std",
    "time_to_90_mean",
    "time_to_90_std",
    "time_to_99_mean",
    "time_to_99_std",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return obj


def config_family(config: str) -> str:
    return re.sub(r"_seed\d+$", "", config)


def run_group(obj: Dict[str, Any]) -> str:
    return "seeded" if obj.get("seed", "") != "" else "single"


def loss_type(obj: Dict[str, Any]) -> str:
    weights = obj.get("loss_weights") or {}
    prompt = float(weights.get("prompt_loss_weight", 1.0))
    answer = float(weights.get("answer_loss_weight", 1.0))
    eos = float(weights.get("eos_loss_weight", prompt))
    if prompt == 0.0 and answer > 0.0 and eos == 0.0:
        return "answer_only"
    if prompt == answer == eos:
        return "full_sequence"
    return f"weighted_prompt={prompt:g}_answer={answer:g}_eos={eos:g}"


def time_to_threshold(history: Iterable[Dict[str, Any]], threshold: float) -> int | None:
    for record in history:
        if float(record.get("eval_exact_match_acc", -math.inf)) >= threshold:
            return int(record["step"])
    return None


def train_row(path: Path, obj: Dict[str, Any]) -> Dict[str, Any]:
    config = str(obj.get("config", path.stem))
    history = obj.get("history") or []
    return {
        "path": str(path),
        "backend": obj.get("backend", ""),
        "config": config,
        "config_family": config_family(config),
        "run_group": run_group(obj),
        "seed": obj.get("seed", ""),
        "device": obj.get("device", ""),
        "model_params": obj.get("model_params", ""),
        "batch_size": obj.get("batch_size", ""),
        "seq_len": obj.get("seq_len", ""),
        "operand_digits": obj.get("operand_digits", ""),
        "result_digits": obj.get("result_digits", ""),
        "answer_order": obj.get("answer_order", ""),
        "loss_type": loss_type(obj),
        "train_steps": obj.get("train_steps", ""),
        "first_step_time_sec": obj.get("first_step_time_sec", ""),
        "steady_state_step_time_ms": obj.get("steady_state_step_time_ms", ""),
        "tokens_per_sec": obj.get("tokens_per_sec", ""),
        "eval_exact_match_accuracy": obj.get("eval_exact_match_accuracy", ""),
        "carry_heavy_exact_match_accuracy": obj.get("carry_heavy_exact_match_accuracy", ""),
        "time_to_50": time_to_threshold(history, 0.50),
        "time_to_90": time_to_threshold(history, 0.90),
        "time_to_99": time_to_threshold(history, 0.99),
    }


def numeric_values(rows: List[Dict[str, Any]], field: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in ("", None):
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            values.append(value_f)
    return values


def mean(values: List[float]) -> str:
    return "" if not values else f"{statistics.fmean(values):.6g}"


def std(values: List[float]) -> str:
    if len(values) < 2:
        return ""
    return f"{statistics.stdev(values):.6g}"


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["backend"], row["config_family"], row["run_group"]), []).append(row)

    out = []
    for (backend, family, group_name), group in sorted(groups.items()):
        seeds = sorted({str(row.get("seed", "")) for row in group if row.get("seed", "") != ""})
        first = group[0]
        agg = {
            "backend": backend,
            "config_family": family,
            "run_group": group_name,
            "n_runs": len(group),
            "seeds": " ".join(seeds),
            "operand_digits": first.get("operand_digits", ""),
            "answer_order": first.get("answer_order", ""),
            "loss_type": first.get("loss_type", ""),
            "train_steps": first.get("train_steps", ""),
        }
        for field in [
            "eval_exact_match_accuracy",
            "carry_heavy_exact_match_accuracy",
            "steady_state_step_time_ms",
            "tokens_per_sec",
            "time_to_90",
            "time_to_99",
        ]:
            values = numeric_values(group, field)
            agg[f"{field}_mean"] = mean(values)
            agg[f"{field}_std"] = std(values)
        out.append(agg)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|")


def write_markdown(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_cell(row.get(field, "")) for field in fields) + " |\n")


def summarize(results_dir: Path) -> None:
    train_paths = sorted(results_dir.glob("train_*.json"))
    rows = [train_row(path, load_json(path)) for path in train_paths]
    agg = aggregate_rows(rows)

    write_csv(results_dir / "summary_table.csv", rows, TRAIN_FIELDS)
    write_markdown(results_dir / "summary_table.md", rows, TRAIN_FIELDS)
    write_csv(results_dir / "summary_aggregate.csv", agg, AGG_FIELDS)
    write_markdown(results_dir / "summary_aggregate.md", agg, AGG_FIELDS)

    print(f"[summarize_results] summarized {len(rows)} training files from {results_dir}")
    print(f"[summarize_results] wrote {results_dir / 'summary_table.csv'}")
    print(f"[summarize_results] wrote {results_dir / 'summary_aggregate.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="Directory containing train_*.json files")
    args = parser.parse_args()
    summarize(Path(args.results_dir))


if __name__ == "__main__":
    main()
