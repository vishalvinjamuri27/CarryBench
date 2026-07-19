"""Create publication-ready plots from CarryBench summary artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_accuracy(results_dir: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _rows(results_dir / "summary_aggregate.csv")
    metric = "test_generated_exact_match_accuracy_mean"
    rows = [row for row in rows if row.get(metric)]
    if not rows:
        return
    labels = [f"{row['backend']}\n{row['config_family']}" for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.7), 5))
    ax.bar(labels, values, color=["#5b8ff9" if row["backend"] == "jax" else "#f6bd16" for row in rows])
    ax.set_ylabel("Generated exact-match accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("CarryBench free-running answer accuracy")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "generated_accuracy.svg")
    fig.savefig(output_dir / "generated_accuracy.png", dpi=180)
    plt.close(fig)


def plot_kv_cache(results_dir: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    path = results_dir / "benchmark_kv_cache_summary.csv"
    if not path.exists():
        return
    rows = _rows(path)
    throughput_field = (
        "steady_decode_tokens_per_sec"
        if rows and "steady_decode_tokens_per_sec" in rows[0]
        else "decode_tokens_per_sec"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, batch_size in sorted({(row["decode_mode"], row["batch_size"]) for row in rows}):
        group = [row for row in rows if row["decode_mode"] == mode and row["batch_size"] == batch_size]
        group.sort(key=lambda row: int(row["generated_tokens"]))
        ax.plot(
            [int(row["generated_tokens"]) for row in group],
            [float(row[throughput_field]) for row in group],
            marker="o",
            label=f"{mode}, batch={batch_size}",
        )
    ax.set_xlabel("Generated tokens")
    ax.set_ylabel("Steady decode tokens/sec")
    ax.set_title("Naive decoding vs KV cache")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "kv_cache_throughput.svg")
    fig.savefig(output_dir / "kv_cache_throughput.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(args.results_dir, args.output_dir)
    plot_kv_cache(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
