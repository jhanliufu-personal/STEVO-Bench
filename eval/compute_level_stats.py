# eval/compute_level_stats.py
"""
Post-pipeline script: computes per-level breakdown statistics and writes them
back into the by_level and overall sections of summary.json.

Run after the eval pipeline:
    python -m eval.compute_level_stats --run_dir runs/my_run

Stats added to each level in by_level, and to overall:
  num_tasks             : number of tasks in the group
  avg_occlusion_done    : fraction of tasks where occlusion was successfully applied
  avg_trigger_applied   : fraction of tasks where the trigger was applied
  avg_acc_baseline      : accuracy on _00 (full-observability baseline) tasks
                          where trigger_applied=True
  avg_acc_full_obs      : accuracy on all fully-observed tasks where trigger_applied=True —
                          baseline (_00) + non-baseline where occlusion_done=False
  avg_acc_occluded      : accuracy on non-baseline tasks where both
                          occlusion_done=True and trigger_applied=True
  avg_success           : fraction of tasks where accuracy == 1.0
  avg_compliant         : fraction of tasks where occlusion_done=True AND trigger_applied=True
  avg_success_compliant : fraction of compliant tasks where accuracy == 1.0

Tasks missing occlusion_done / trigger_applied (not yet control-judged) are
excluded from the relevant averages but still counted in avg_acc_full_obs if
occlusion clearly did not happen (occlusion_done absent treated as False).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _avg(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _empty_buckets() -> Dict[str, List]:
    return {
        "occlusion_done": [],
        "trigger_applied": [],
        "acc_baseline": [],
        "acc_full_obs": [],
        "acc_occluded": [],
        "success": [],
        "compliant": [],
        "success_compliant": [],
    }


def _fill_bucket(bucket: Dict[str, List], t: Dict[str, Any]) -> None:
    """Accumulate one task entry into a bucket dict."""
    task_id = str(t.get("task_id", ""))
    accuracy = t.get("accuracy")
    occlusion_done = t.get("occlusion_done")   # bool or None
    trigger_applied = t.get("trigger_applied")  # bool or None
    is_baseline = task_id.endswith("_00")

    if occlusion_done is not None:
        bucket["occlusion_done"].append(1.0 if occlusion_done else 0.0)
    if trigger_applied is not None:
        bucket["trigger_applied"].append(1.0 if trigger_applied else 0.0)

    compliant = bool(occlusion_done) and bool(trigger_applied)
    if occlusion_done is not None and trigger_applied is not None:
        bucket["compliant"].append(1.0 if compliant else 0.0)

    if accuracy is None:
        return

    success = 1.0 if accuracy == 1.0 else 0.0
    bucket["success"].append(success)

    if compliant:
        bucket["success_compliant"].append(success)

    if is_baseline and trigger_applied:
        bucket["acc_baseline"].append(accuracy)

    if (is_baseline or not occlusion_done) and trigger_applied:
        bucket["acc_full_obs"].append(accuracy)

    if not is_baseline and occlusion_done and trigger_applied:
        bucket["acc_occluded"].append(accuracy)


def _bucket_to_stats(bucket: Dict[str, List], num_tasks: int) -> Dict[str, Any]:
    return {
        "num_tasks":              num_tasks,
        "avg_occlusion_done":     _avg(bucket["occlusion_done"]),
        "avg_trigger_applied":    _avg(bucket["trigger_applied"]),
        "avg_acc_baseline":       _avg(bucket["acc_baseline"]),
        "avg_acc_full_obs":       _avg(bucket["acc_full_obs"]),
        "avg_acc_occluded":       _avg(bucket["acc_occluded"]),
        "avg_success":            _avg(bucket["success"]),
        "avg_compliant":          _avg(bucket["compliant"]),
        "avg_success_compliant":  _avg(bucket["success_compliant"]),
    }


def compute_level_stats(
    tasks: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Compute per-level and overall breakdown stats.

    Returns:
      (level_stats, overall_stats)
      level_stats  — dict keyed by level string
      overall_stats — same seven metrics aggregated across all levels
    """
    level_buckets: Dict[str, Dict[str, List]] = defaultdict(_empty_buckets)
    level_counts:  Dict[str, int] = defaultdict(int)
    overall_bucket = _empty_buckets()
    overall_count = 0

    for t in tasks:
        level = str(t.get("task_level", ""))
        if not level:
            continue
        _fill_bucket(level_buckets[level], t)
        level_counts[level] += 1
        _fill_bucket(overall_bucket, t)
        overall_count += 1

    level_stats = {
        level: _bucket_to_stats(level_buckets[level], level_counts[level])
        for level in level_buckets
    }
    overall_stats = _bucket_to_stats(overall_bucket, overall_count)

    return level_stats, overall_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-level breakdown stats and write them into summary.json."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        type=str,
        help="Path to the eval run directory containing summary.json.",
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

    level_stats, overall_stats = compute_level_stats(tasks)

    # Update by_level
    by_level = summary.setdefault("by_level", {})
    for level, stats in level_stats.items():
        if level not in by_level:
            by_level[level] = {}
        by_level[level].update(stats)

    # Update overall
    summary.setdefault("overall", {}).update(overall_stats)

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print summary table
    print(f"Updated {summary_path}\n")

    def fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, float) else f"{v:>5}" if isinstance(v, int) else "  n/a"

    header = (
        f"{'Level':>7}  {'N':>5}  {'occ_done':>8}  {'trig_app':>8}  "
        f"{'baseline':>8}  {'full_obs':>8}  {'occluded':>8}  "
        f"{'success':>8}  {'compliant':>9}  {'succ_cmp':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = sorted(by_level.keys(), key=lambda x: int(x) if x.isdigit() else x)
    for level in rows:
        s = by_level[level]
        print(
            f"{level:>7}  "
            f"{s.get('num_tasks', '?'):>5}  "
            f"{fmt(s.get('avg_occlusion_done')):>8}  "
            f"{fmt(s.get('avg_trigger_applied')):>8}  "
            f"{fmt(s.get('avg_acc_baseline')):>8}  "
            f"{fmt(s.get('avg_acc_full_obs')):>8}  "
            f"{fmt(s.get('avg_acc_occluded')):>8}  "
            f"{fmt(s.get('avg_success')):>8}  "
            f"{fmt(s.get('avg_compliant')):>9}  "
            f"{fmt(s.get('avg_success_compliant')):>8}"
        )

    print("-" * len(header))
    o = summary["overall"]
    print(
        f"{'overall':>7}  "
        f"{o.get('num_tasks', '?'):>5}  "
        f"{fmt(o.get('avg_occlusion_done')):>8}  "
        f"{fmt(o.get('avg_trigger_applied')):>8}  "
        f"{fmt(o.get('avg_acc_baseline')):>8}  "
        f"{fmt(o.get('avg_acc_full_obs')):>8}  "
        f"{fmt(o.get('avg_acc_occluded')):>8}  "
        f"{fmt(o.get('avg_success')):>8}  "
        f"{fmt(o.get('avg_compliant')):>9}  "
        f"{fmt(o.get('avg_success_compliant')):>8}"
    )


if __name__ == "__main__":
    main()
