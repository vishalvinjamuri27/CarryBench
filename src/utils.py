"""Small shared helpers: config loading, seeding, timing, result I/O."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
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
    return cfg


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
