#!/usr/bin/env python3
# eval/plot_confusion_matrix.py
"""
Plot a confusion matrix between the LLM judge and human judge for a given metric.

Usage:
    python -m eval.plot_confusion_matrix --run_dir runs/my_run --metric occlusion_done
    python -m eval.plot_confusion_matrix --run_dir runs/my_run --metric artifact --level 3
    python -m eval.plot_confusion_matrix --run_dir runs/my_run --metric state_evol --output cm.png

Supported metrics  (LLM value source             → human value source)
  occlusion_done   task["occlusion_done"]          task["human_occlusion_done"]
  trigger_applied  task["trigger_applied"]         task["human_trigger_applied"]
  artifact         task["artifact"]                task["human_artifact"]
  state_evol       task["accuracy"] == 1.0         task["human_state_evol"]

Only tasks where both the LLM value and the human value are non-null are included.

Outputs:
  - Text summary (agreement rate, Cohen's κ, per-cell counts) to stdout
  - PNG confusion matrix saved to <run_dir>/confusion_<metric>[_level<N>].png
    (override with --output)
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Supported metrics
# ---------------------------------------------------------------------------

SUPPORTED_METRICS = ["occlusion_done", "trigger_applied", "artifact", "state_evol"]


def _get_llm_value(task: dict, metric: str) -> Optional[bool]:
    """Return the LLM judge boolean for a task entry, or None if absent."""
    if metric == "state_evol":
        acc = task.get("accuracy")
        return (acc == 1.0) if acc is not None else None
    val = task.get(metric)
    return bool(val) if val is not None else None


def _get_human_value(task: dict, metric: str) -> Optional[bool]:
    """Return the human judge boolean for a task entry, or None if absent."""
    key = "human_state_evol" if metric == "state_evol" else f"human_{metric}"
    val = task.get(key)
    return bool(val) if val is not None else None


# ---------------------------------------------------------------------------
# Confusion matrix  (rows = Human ground truth, cols = LLM prediction)
#
#              LLM=T   LLM=F
#  Human=T  |   TP   |   FN  |
#  Human=F  |   FP   |   TN  |
# ---------------------------------------------------------------------------

def compute_confusion(
    tasks: List[dict],
    metric: str,
    level: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Returns:
        cm : 2×2 int ndarray  —  cm[human_idx, llm_idx]
             row/col 0 = True, row/col 1 = False
        n  : number of tasks included
    """
    cm = np.zeros((2, 2), dtype=int)
    included = 0
    for t in tasks:
        if level is not None and str(t.get("task_level", "")) != level:
            continue
        llm   = _get_llm_value(t, metric)
        human = _get_human_value(t, metric)
        if llm is None or human is None:
            continue
        row = 0 if human else 1
        col = 0 if llm   else 1
        cm[row, col] += 1
        included += 1
    return cm, included


def _cohen_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = (cm[0, 0] + cm[1, 1]) / n
    pe = (
        (cm[0, :].sum() * cm[:, 0].sum() + cm[1, :].sum() * cm[:, 1].sum())
        / (n * n)
    )
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    metric: str,
    n: int,
    output_path: Path,
    level: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    agreement = (cm[0, 0] + cm[1, 1]) / n if n > 0 else float("nan")
    kappa = _cohen_kappa(cm)

    # Cell colours: diagonal (TP/TN) = green tint, off-diagonal = red tint
    cell_colors = [
        ["#1a3a1a", "#3a1a1a"],  # Human=T row
        ["#3a1a1a", "#1a3a1a"],  # Human=F row
    ]
    labels = ["True (T)", "False (F)"]

    fig, ax = plt.subplots(figsize=(5, 4.8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=cell_colors[i][j], zorder=0,
            ))
            count = cm[i, j]
            pct   = 100.0 * count / n if n > 0 else 0.0
            ax.text(
                j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="#c9d1d9", zorder=1,
            )

    # Grid
    for v in [-0.5, 0.5, 1.5]:
        ax.axhline(v, color="#30363d", linewidth=1.2, zorder=2)
        ax.axvline(v, color="#30363d", linewidth=1.2, zorder=2)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()   # row 0 (Human=T) at top

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="#c9d1d9", fontsize=11)
    ax.set_yticklabels(labels, color="#c9d1d9", fontsize=11)
    ax.set_xlabel("LLM Judge", color="#58a6ff", fontsize=12, labelpad=8)
    ax.set_ylabel("Human Judge", color="#58a6ff", fontsize=12, labelpad=8)
    ax.tick_params(colors="#8b949e", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    level_str = f" · Level {level}" if level else ""
    if title:
        main_title = title
    else:
        main_title = f"Confusion Matrix: {metric}{level_str}"
    subtitle = f"N = {n}  |  Agreement = {agreement:.1%}  |  κ = {kappa:.3f}"

    fig.suptitle(main_title, color="#c9d1d9", fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(subtitle, color="#8b949e", fontsize=10, pad=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot LLM-vs-human confusion matrix for a given metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run_dir", required=True,
        help="Eval run directory containing summary.json.",
    )
    parser.add_argument(
        "--metric", required=True, choices=SUPPORTED_METRICS,
        help="Metric name. One of: " + ", ".join(SUPPORTED_METRICS),
    )
    parser.add_argument(
        "--level", default=None,
        help="Restrict to a specific task level (e.g. 1, 2, …). Default: all levels.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path. Default: <run_dir>/confusion_<metric>[_level<N>].png",
    )
    parser.add_argument(
        "--title", default=None,
        help="Custom plot title (overrides the auto-generated title).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = summary.get("tasks", [])
    if not tasks:
        raise ValueError("summary.json contains no tasks.")

    cm, n = compute_confusion(tasks, args.metric, level=args.level)

    if n == 0:
        level_str = f" at level {args.level}" if args.level else ""
        print(
            f"No tasks{level_str} have both LLM and human values for "
            f"'{args.metric}'. Nothing to plot."
        )
        return

    # Text summary
    agree = (cm[0, 0] + cm[1, 1]) / n
    level_str = f"  (level {args.level})" if args.level else ""
    print(f"\nConfusion matrix: {args.metric}{level_str}")
    print(f"  N = {n}")
    print(f"  {'':16s}  LLM=T   LLM=F")
    print(f"  {'Human=T':16s}  {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"  {'Human=F':16s}  {cm[1, 0]:5d}   {cm[1, 1]:5d}")
    print(f"  Agreement : {agree:.1%}")
    print(f"  Cohen's κ : {_cohen_kappa(cm):.3f}")

    # Output path
    level_suffix = f"_level{args.level}" if args.level else ""
    default_out  = run_dir / f"confusion_{args.metric}{level_suffix}.png"
    output_path  = Path(args.output) if args.output else default_out

    plot_confusion_matrix(
        cm, args.metric, n, output_path,
        level=args.level, title=args.title,
    )


if __name__ == "__main__":
    main()
