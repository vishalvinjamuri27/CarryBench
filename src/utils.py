"""Small shared helpers: config loading, seeding, timing, result I/O."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file {path} is empty")
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Fail early on malformed experiment configurations."""
    required = {
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "train_steps",
        "eval_every",
        "n_train",
        "n_eval",
        "n_carry_heavy",
    }
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")
    positive = required - {"weight_decay", "learning_rate"}
    for key in positive:
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive, got {cfg[key]!r}")
    if float(cfg["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if float(cfg["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative")
    if int(cfg["d_model"]) % int(cfg["n_heads"]) != 0:
        raise ValueError("d_model must be divisible by n_heads")
    for key in ("prompt_loss_weight", "answer_loss_weight", "eos_loss_weight"):
        if float(cfg.get(key, 1.0)) < 0:
            raise ValueError(f"{key} must be non-negative")
    if cfg.get("precision", "float32") not in {"float32", "bfloat16"}:
        raise ValueError("precision must be 'float32' or 'bfloat16'")


@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0


def model_config_from_dict(cfg: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg.get("dropout", 0.0),
    )


@contextmanager
def timer():
    """Context manager yielding elapsed wall-clock seconds via a mutable list.

    Usage:
        with timer() as t:
            do_work()
        print(t[0])  # seconds
    """
    box = [0.0]
    start = time.perf_counter()
    yield box
    box[0] = time.perf_counter() - start


def save_json(obj: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def append_csv_row(row: Dict[str, Any], path: str) -> None:
    """Append a single row of metrics to a CSV, writing a header if new."""
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def count_params(params_pytree) -> int:
    """Count total scalar parameters in a JAX/Flax pytree of arrays."""
    import jax

    return sum(x.size for x in jax.tree_util.tree_leaves(params_pytree))


def environment_metadata() -> Dict[str, Any]:
    """Collect enough version and revision data to audit an experiment."""
    versions: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("jax", "jaxlib", "flax", "optax", "torch", "numpy"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = None
    try:
        versions["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        versions["git_commit"] = None
    return versions


def timing_statistics(step_times: list[float]) -> Dict[str, float]:
    """Summarize repeated synchronized step timings in milliseconds."""
    import numpy as np

    if not step_times:
        return {key: float("nan") for key in ("mean_ms", "median_ms", "p95_ms", "std_ms")}
    values = np.asarray(step_times, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "std_ms": float(np.std(values)),
    }
