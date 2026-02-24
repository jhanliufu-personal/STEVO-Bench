# eval/score_task.py
"""
score_task.py

Scoring logic for output-first evaluation.

Assumption (Option A):
- All binary questions are phrased so that a correct final frame yields answer == "yes".
- Therefore:
    yes      -> correct
    no       -> incorrect
    unknown  -> incorrect (but tracked separately)

This module is intentionally simple and transparent.
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
from collections import defaultdict

from eval.judge_output_parser import JudgeResult, JudgeAnswer


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class TaskScore:
    task_id: str
    task_level: int

    num_questions: int
    num_yes: int
    num_no: int
    num_unknown: int

    accuracy: float  # num_yes / num_questions

    provider: str
    model: str
    report_path: str


# -----------------------------
# Scoring
# -----------------------------

def score_one_task(judge_result: JudgeResult) -> TaskScore:
    """
    Score a single task.

    YES      -> correct
    NO       -> incorrect
    UNKNOWN  -> incorrect (but counted separately)

    Returns TaskScore.
    """
    answers: List[JudgeAnswer] = judge_result.answers

    num_questions = len(answers)
    num_yes = sum(1 for a in answers if a.answer == "yes")
    num_no = sum(1 for a in answers if a.answer == "no")
    num_unknown = sum(1 for a in answers if a.answer == "unknown")

    accuracy = (num_yes / num_questions) if num_questions > 0 else 0.0

    return TaskScore(
        task_id=judge_result.task_id,
        task_level=judge_result.task_level,
        num_questions=num_questions,
        num_yes=num_yes,
        num_no=num_no,
        num_unknown=num_unknown,
        accuracy=accuracy,
        provider=judge_result.provider,
        model=judge_result.model,
        report_path=judge_result.report_path,
    )


def score_all_tasks(judge_results: List[JudgeResult]) -> Dict[str, TaskScore]:
    """
    Score all tasks and append score info to each task's judge_report.json.

    Appended JSON structure:

      "score": {
        "accuracy": 0.875,
        "num_questions": 8,
        "num_yes": 7,
        "num_no": 1,
        "num_unknown": 0
      }

    Returns:
      dict mapping task_id -> TaskScore
    """
    scores: Dict[str, TaskScore] = {}

    for jr in judge_results:
        task_score = score_one_task(jr)
        scores[jr.task_id] = task_score

        report_path = Path(jr.report_path)
        if not report_path.exists():
            raise FileNotFoundError(f"judge_report.json not found: {report_path}")

        report = json.loads(report_path.read_text(encoding="utf-8"))

        # Append (or overwrite) score block
        report["score"] = {
            "accuracy": task_score.accuracy,
            "num_questions": task_score.num_questions,
            "num_yes": task_score.num_yes,
            "num_no": task_score.num_no,
            "num_unknown": task_score.num_unknown,
        }

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return scores


def write_run_report(task_scores: Dict[str, TaskScore], run_dir: Path) -> Dict[str, Any]:
    """
    Merge quality scores into summary.json (creates it if absent).

    Existing fields written by other pipeline stages (e.g. control judge fields
    occlusion_done, trigger_applied, artifact) are preserved in each task entry.

    Inputs:
      task_scores: dict mapping task_id -> TaskScore
      run_dir: run root directory

    Outputs:
      - prints overall / per-level / per-task accuracy tables
      - saves / updates <run_dir>/summary.json

    Returns:
      report dict (same content as saved JSON)
    """
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    # Load existing summary to preserve fields from other pipeline stages
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        existing = {"num_tasks": 0, "overall": {}, "by_level": {}, "tasks": []}

    if not task_scores:
        summary_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print("[WARN] No task scores provided.")
        return existing

    scores = list(task_scores.values())

    # Overall accuracy (over scored tasks only)
    overall_avg = sum(ts.accuracy for ts in scores) / len(scores)

    # By level — merge into existing by_level to preserve other per-level stats
    level_buckets: Dict[int, list] = defaultdict(list)
    for ts in scores:
        level_buckets[int(ts.task_level)].append(ts)

    by_level = existing.get("by_level") or {}
    for lvl, items in sorted(level_buckets.items()):
        by_level.setdefault(str(lvl), {}).update({
            "num_tasks": len(items),
            "avg_accuracy": sum(t.accuracy for t in items) / len(items),
        })

    # Merge quality fields into existing task entries (preserves control fields)
    task_entry_map: Dict[str, dict] = {
        t["task_id"]: t
        for t in existing.get("tasks", [])
        if "task_id" in t
    }
    for task_id, ts in task_scores.items():
        entry = task_entry_map.setdefault(task_id, {"task_id": task_id})
        entry.update({
            "task_level":    int(ts.task_level),
            "accuracy":      ts.accuracy,
            "num_questions": ts.num_questions,
            "num_yes":       ts.num_yes,
            "num_no":        ts.num_no,
            "num_unknown":   ts.num_unknown,
            "provider":      ts.provider,
            "model":         ts.model,
            "report_path":   ts.report_path,
        })

    task_rows = sorted(
        task_entry_map.values(),
        key=lambda x: (x.get("task_level", 0), x.get("task_id", "")),
    )

    report = {
        "num_tasks": len(task_rows),
        "overall":   {**existing.get("overall", {}), "avg_accuracy": overall_avg},
        "by_level":  by_level,
        "tasks":     task_rows,
    }

    # Print summary
    print("\n=== Run Summary ===")
    print(f"Num tasks: {len(scores)}")
    print(f"Overall avg accuracy: {overall_avg:.4f}")

    print("\n=== Avg Accuracy by Level ===")
    for lvl_str, stats in sorted(by_level.items()):
        if "avg_accuracy" in stats:
            print(f"Level {lvl_str}: avg={stats['avg_accuracy']:.4f} (n={stats.get('num_tasks', '?')})")

    print("\n=== Per-Task Scores ===")
    header = f"{'level':>5}  {'accuracy':>8}  {'yes':>4}  {'no':>3}  {'unk':>4}  {'nq':>3}  task_id"
    print(header)
    print("-" * len(header))
    for r in task_rows:
        if "accuracy" not in r:
            continue
        print(
            f"{r['task_level']:>5}  {r['accuracy']:>8.4f}  {r['num_yes']:>4}  {r['num_no']:>3}  "
            f"{r['num_unknown']:>4}  {r['num_questions']:>3}  {r['task_id']}"
        )

    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved run summary to: {summary_path}")

    return report
