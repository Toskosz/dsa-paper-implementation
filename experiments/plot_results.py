from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Row:
    timestamp: int
    phase: str
    num_ues: int
    num_edges: int
    exec_time_ms: float
    prb_success_rate: float
    fairness: float


def load_csv(path: str) -> list[Row]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(Row(
                timestamp=int(r["timestamp"]),
                phase=r["phase"],
                num_ues=int(r["num_ues"]),
                num_edges=int(r["num_edges"]),
                exec_time_ms=float(r["exec_time_ms"]),
                prb_success_rate=float(r["prb_success_rate"]),
                fairness=float(r["fairness"]),
            ))
    return rows


PHASE_COLORS = {"average": "blue", "worst": "red", "best": "green"}
PHASE_LABELS = {
    "average": "Phase 1: Average Case O(U log U + E)",
    "worst": "Phase 2: Worst Case O(U²)",
    "best": "Phase 3: Best Case Ω(U log U)",
}


def plot_exec_time_vs_ues(rows: list[Row], output: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for phase, color in PHASE_COLORS.items():
        data = [(r.num_ues, r.exec_time_ms) for r in rows if r.phase == phase]
        if not data:
            continue
        data.sort(key=lambda x: x[0])
        ues, times = zip(*data)
        ax.scatter(ues, times, c=color, alpha=0.4, s=8, label=PHASE_LABELS[phase])

    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Graph Coloring Execution Time (ms)")
    ax.set_title("Execution Time vs. Number of UEs by Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_edges_vs_ues(rows: list[Row], output: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for phase, color in PHASE_COLORS.items():
        data = [(r.num_ues, r.num_edges) for r in rows if r.phase == phase]
        if not data:
            continue
        data.sort(key=lambda x: x[0])
        ues, edges = zip(*data)
        ax.scatter(ues, edges, c=color, alpha=0.4, s=8, label=PHASE_LABELS[phase])

    ax.set_xlabel("Number of UEs")
    ax.set_ylabel("Number of Conflict Edges")
    ax.set_title("Conflict Edges vs. Number of UEs by Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_prb_success_rate(rows: list[Row], output: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for phase, color in PHASE_COLORS.items():
        data = [(r.timestamp, r.prb_success_rate) for r in rows if r.phase == phase]
        if not data:
            continue
        ts, rates = zip(*data)
        ax.plot(ts, rates, c=color, alpha=0.7, linewidth=1, label=PHASE_LABELS[phase])

    ax.axhline(y=0.9, color="black", linestyle="--", alpha=0.5, label="Target 90%")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("PRB Assignment Success Rate")
    ax.set_title("PRB Assignment Success Rate Over Time")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_fairness(rows: list[Row], output: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for phase, color in PHASE_COLORS.items():
        data = [(r.timestamp, r.fairness) for r in rows if r.phase == phase]
        if not data:
            continue
        ts, fairness = zip(*data)
        ax.plot(ts, fairness, c=color, alpha=0.7, linewidth=1, label=PHASE_LABELS[phase])

    ax.axhline(y=0.85, color="black", linestyle="--", alpha=0.5, label="Target 85%")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Service-Share Fairness (Jain's Index)")
    ax.set_title("Fairness Over Time")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_complexity_summary(rows: list[Row], output: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, phase in zip(axes, ["average", "worst", "best"]):
        data = [(r.num_ues, r.exec_time_ms) for r in rows if r.phase == phase]
        if not data:
            ax.set_title(PHASE_LABELS[phase])
            continue
        data.sort(key=lambda x: x[0])
        ues, times = zip(*data)
        ues_arr = np.array(ues, dtype=float)
        times_arr = np.array(times, dtype=float)

        ax.scatter(ues_arr, times_arr, c="gray", alpha=0.3, s=6)

        n_bins = min(30, len(ues_arr) // 5) if len(ues_arr) > 30 else len(ues_arr)
        if n_bins > 0:
            bin_indices = np.array_split(np.argsort(ues_arr), n_bins)
            bin_ues = [ues_arr[idx].mean() for idx in bin_indices]
            bin_times = [times_arr[idx].mean() for idx in bin_indices]
            ax.plot(bin_ues, bin_times, "r-", linewidth=2, label="Binned mean")

        ax.set_xlabel("Number of UEs")
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title(PHASE_LABELS[phase])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simulation results")
    parser.add_argument("--csv", default="results/simulation_log.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    plot_exec_time_vs_ues(rows, f"{args.output_dir}/plot_exec_time_vs_ues.png")
    plot_edges_vs_ues(rows, f"{args.output_dir}/plot_edges_vs_ues.png")
    plot_prb_success_rate(rows, f"{args.output_dir}/plot_prb_success_rate.png")
    plot_fairness(rows, f"{args.output_dir}/plot_fairness.png")
    plot_complexity_summary(rows, f"{args.output_dir}/plot_complexity_summary.png")
    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
