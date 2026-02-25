# eval/compute_level_stats.py
"""
Post-pipeline script: computes per-level breakdown statistics and writes them
back into the by_level and overall sections of summary.json.

Run after the eval pipeline:
    python -m eval.compute_level_stats --run_dir runs/my_run

Stats added to each level in by_level, and to overall:
  num_tasks              : number of tasks in the group
  avg_occlusion_done     : fraction of tasks where occlusion was successfully applied
  avg_trigger_applied    : fraction of tasks where the trigger was applied
  avg_artifact           : fraction of tasks where obvious visual artifacts are present
  avg_control_success    : fraction of tasks where occlusion_done=True AND trigger_applied=True
  avg_state_evol_success : fraction of tasks where accuracy == 1.0
  avg_task_success       : fraction of tasks where control_success=True AND
                           state_evol_success=True AND artifact=False

Tasks missing occlusion_done / trigger_applied / artifact (not yet control-judged)
are excluded from the relevant averages.
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
        # LLM judge metrics
        "occlusion_done":     [],
        "trigger_applied":    [],
        "artifact":           [],
        "control_success":    [],
        "state_evol_success": [],
        "task_success":       [],
        # Human judge metrics
        "human_occlusion_done":     [],
        "human_trigger_applied":    [],
        "human_artifact":           [],
        "human_control_success":    [],
        "human_state_evol_success": [],
        "human_task_success":       [],
        "human_final_frame_correct": [],
        "human_coherence":           [],
        # Human metrics conditioned on human control success
        # (populated only for tasks where human_occlusion_done=True AND human_trigger_applied=True)
        "human_ctrl_n":                   [],  # counter: 1 per ctrl-success task
        "human_state_evol_ctrl":          [],
        "human_final_frame_correct_ctrl": [],
        "human_artifact_ctrl":            [],
        "human_coherence_ctrl":           [],
    }


def _fill_bucket(bucket: Dict[str, List], t: Dict[str, Any]) -> None:
    """Accumulate one task entry into a bucket dict."""
    accuracy        = t.get("accuracy")
    occlusion_done  = t.get("occlusion_done")   # bool or None
    trigger_applied = t.get("trigger_applied")  # bool or None
    artifact        = t.get("artifact")         # bool or None

    if occlusion_done is not None:
        bucket["occlusion_done"].append(1.0 if occlusion_done else 0.0)
    if trigger_applied is not None:
        bucket["trigger_applied"].append(1.0 if trigger_applied else 0.0)
    if artifact is not None:
        bucket["artifact"].append(1.0 if artifact else 0.0)

    # control_success: occlusion_done AND trigger_applied (no artifact check)
    control_success = bool(occlusion_done) and bool(trigger_applied)
    if occlusion_done is not None and trigger_applied is not None:
        bucket["control_success"].append(1.0 if control_success else 0.0)

    # state_evol_success: accuracy == 1.0
    if accuracy is not None:
        bucket["state_evol_success"].append(1.0 if accuracy == 1.0 else 0.0)

    # task_success: control_success AND accuracy == 1.0 AND no artifact
    if accuracy is not None and occlusion_done is not None and trigger_applied is not None and artifact is not None:
        task_success = control_success and (accuracy == 1.0) and not bool(artifact)
        bucket["task_success"].append(1.0 if task_success else 0.0)

    # ---- Human counterparts ----
    human_occlusion_done       = t.get("human_occlusion_done")        # bool or None
    human_trigger_applied      = t.get("human_trigger_applied")       # bool or None
    human_artifact             = t.get("human_artifact")              # bool or None
    human_state_evol           = t.get("human_state_evol")            # bool or None
    human_final_frame_correct  = t.get("human_final_frame_correct")   # bool or None
    human_coherence            = t.get("human_coherence")             # bool or None

    if human_occlusion_done is not None:
        bucket["human_occlusion_done"].append(1.0 if human_occlusion_done else 0.0)
    if human_trigger_applied is not None:
        bucket["human_trigger_applied"].append(1.0 if human_trigger_applied else 0.0)
    if human_artifact is not None:
        bucket["human_artifact"].append(1.0 if human_artifact else 0.0)
    if human_final_frame_correct is not None:
        bucket["human_final_frame_correct"].append(1.0 if human_final_frame_correct else 0.0)
    if human_coherence is not None:
        bucket["human_coherence"].append(1.0 if human_coherence else 0.0)

    # human_control_success: human_occlusion_done AND human_trigger_applied
    human_ctrl = False
    if human_occlusion_done is not None and human_trigger_applied is not None:
        human_ctrl = bool(human_occlusion_done) and bool(human_trigger_applied)
        bucket["human_control_success"].append(1.0 if human_ctrl else 0.0)

    # human_state_evol_success: human_state_evol is True
    if human_state_evol is not None:
        bucket["human_state_evol_success"].append(1.0 if human_state_evol else 0.0)

    # human_task_success: human_ctrl AND human_state_evol AND NOT human_artifact
    if (human_occlusion_done is not None and human_trigger_applied is not None
            and human_artifact is not None and human_state_evol is not None):
        human_task = (bool(human_occlusion_done) and bool(human_trigger_applied)
                      and bool(human_state_evol) and not bool(human_artifact))
        bucket["human_task_success"].append(1.0 if human_task else 0.0)

    # ---- Conditional human metrics (only for ctrl-success tasks) ----
    if human_ctrl:
        bucket["human_ctrl_n"].append(1)
        if human_state_evol is not None:
            bucket["human_state_evol_ctrl"].append(1.0 if human_state_evol else 0.0)
        if human_final_frame_correct is not None:
            bucket["human_final_frame_correct_ctrl"].append(1.0 if human_final_frame_correct else 0.0)
        if human_artifact is not None:
            bucket["human_artifact_ctrl"].append(1.0 if human_artifact else 0.0)
        if human_coherence is not None:
            bucket["human_coherence_ctrl"].append(1.0 if human_coherence else 0.0)


def _bucket_to_stats(bucket: Dict[str, List], num_tasks: int) -> Dict[str, Any]:
    return {
        "num_tasks":               num_tasks,
        # LLM judge averages
        "avg_occlusion_done":      _avg(bucket["occlusion_done"]),
        "avg_trigger_applied":     _avg(bucket["trigger_applied"]),
        "avg_artifact":            _avg(bucket["artifact"]),
        "avg_control_success":     _avg(bucket["control_success"]),
        "avg_state_evol_success":  _avg(bucket["state_evol_success"]),
        "avg_task_success":        _avg(bucket["task_success"]),
        # Human judge averages (over all tasks with valid answers)
        "avg_human_occlusion_done":      _avg(bucket["human_occlusion_done"]),
        "avg_human_trigger_applied":     _avg(bucket["human_trigger_applied"]),
        "avg_human_artifact":            _avg(bucket["human_artifact"]),
        "avg_human_control_success":     _avg(bucket["human_control_success"]),
        "avg_human_state_evol_success":  _avg(bucket["human_state_evol_success"]),
        "avg_human_task_success":        _avg(bucket["human_task_success"]),
        "avg_human_final_frame_correct": _avg(bucket["human_final_frame_correct"]),
        "avg_human_coherence":           _avg(bucket["human_coherence"]),
        # Human metrics conditioned on human control success
        "human_ctrl_n":                       len(bucket["human_ctrl_n"]),
        "avg_human_state_evol_ctrl":          _avg(bucket["human_state_evol_ctrl"]),
        "avg_human_final_frame_correct_ctrl": _avg(bucket["human_final_frame_correct_ctrl"]),
        "avg_human_artifact_ctrl":            _avg(bucket["human_artifact_ctrl"]),
        "avg_human_coherence_ctrl":           _avg(bucket["human_coherence_ctrl"]),
    }


def _stats_for_tasks(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate stats for an arbitrary list of task entries."""
    bucket = _empty_buckets()
    for t in tasks:
        _fill_bucket(bucket, t)
    return _bucket_to_stats(bucket, len(tasks))


def compute_level_stats(
    tasks: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Compute per-level and overall breakdown stats.

    Returns:
      (level_stats, overall_stats)
      level_stats   — dict keyed by level string
      overall_stats — same metrics aggregated across all levels
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

    # Split tasks into baseline (_00) and occluded (non-_00) groups
    baseline_tasks = [t for t in tasks if str(t.get("task_id", "")).endswith("_00")]
    occluded_tasks = [t for t in tasks if not str(t.get("task_id", "")).endswith("_00")]

    all_stats      = _stats_for_tasks(tasks)
    baseline_stats = _stats_for_tasks(baseline_tasks)
    occluded_stats = _stats_for_tasks(occluded_tasks)

    # Print summary tables
    print(f"Updated {summary_path}\n")

    def fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, float) else f"{v:>5}" if isinstance(v, int) else "  n/a"

    def print_table(title: str, llm_key_prefix: str) -> None:
        p = llm_key_prefix  # "" for LLM, "human_" for Human
        header = (
            f"{'Group':>10}  {'N':>5}  {'occ_done':>8}  {'trig_app':>8}  "
            f"{'artifact':>8}  {'ctrl_suc':>8}  {'se_suc':>8}  {'task_suc':>8}"
        )
        print(title)
        print(header)
        print("-" * len(header))
        for label, s in [("all", all_stats), ("baseline", baseline_stats), ("occluded", occluded_stats)]:
            print(
                f"{label:>10}  "
                f"{s.get('num_tasks', '?'):>5}  "
                f"{fmt(s.get(f'avg_{p}occlusion_done')):>8}  "
                f"{fmt(s.get(f'avg_{p}trigger_applied')):>8}  "
                f"{fmt(s.get(f'avg_{p}artifact')):>8}  "
                f"{fmt(s.get(f'avg_{p}control_success')):>8}  "
                f"{fmt(s.get(f'avg_{p}state_evol_success')):>8}  "
                f"{fmt(s.get(f'avg_{p}task_success')):>8}"
            )

    print_table("LLM Judge",   llm_key_prefix="")
    print()
    print_table("Human Judge", llm_key_prefix="human_")
    print()

    def print_human_ctrl_table() -> None:
        """Print human metrics conditioned on control success, one row per group."""
        header = (
            f"{'Group':>10}  {'N':>5}  {'N_ctrl':>6}  {'ctrl_suc':>8}  "
            f"{'se_suc|c':>8}  {'ffc|c':>8}  {'artif|c':>8}  {'coher|c':>8}"
        )
        print("Human Metrics | state/coherence evaluated given human control success")
        print("  (se_suc|c, ffc|c, artif|c, coher|c computed over ctrl-success tasks only)")
        print(header)
        print("-" * len(header))
        for label, s in [("all", all_stats), ("baseline", baseline_stats), ("occluded", occluded_stats)]:
            print(
                f"{label:>10}  "
                f"{s.get('num_tasks', '?'):>5}  "
                f"{fmt(s.get('human_ctrl_n', 0)):>6}  "
                f"{fmt(s.get('avg_human_control_success')):>8}  "
                f"{fmt(s.get('avg_human_state_evol_ctrl')):>8}  "
                f"{fmt(s.get('avg_human_final_frame_correct_ctrl')):>8}  "
                f"{fmt(s.get('avg_human_artifact_ctrl')):>8}  "
                f"{fmt(s.get('avg_human_coherence_ctrl')):>8}"
            )

    print_human_ctrl_table()


if __name__ == "__main__":
    main()
