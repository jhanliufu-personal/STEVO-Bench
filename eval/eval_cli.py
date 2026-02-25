import argparse
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from eval.task_resolver import resolve_tasks_from_outputs_json, ResolvedTask
from eval.judge_runner import judge_one_task
from eval.score_task import score_all_tasks, write_run_report
from eval.control_judge import (
    evaluate_control_one_task, append_control_results_to_summary,
    _load_task_fields, _compute_requested_fields,
)
from eval.judge_output_parser import JudgeResult
from eval.control_judge import ControlJudgeResult


def _init_summary_json(resolved_tasks: List[ResolvedTask], run_root: Path) -> None:
    """
    Ensure summary.json exists with a skeleton entry for every resolved task.

    - If the file does not exist, create it with num_tasks / overall / by_level / tasks.
    - If the file already exists, add entries for any task_ids not yet present
      (handles re-runs with --pattern or newly added tasks) without touching
      existing entries.
    """
    summary_path = run_root / "summary.json"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_ids = {t["task_id"] for t in summary.get("tasks", []) if "task_id" in t}
        new_entries = [
            {"task_id": t.task_id, "task_level": t.task_level}
            for t in resolved_tasks
            if t.task_id not in existing_ids
        ]
        if new_entries:
            summary.setdefault("tasks", []).extend(new_entries)
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    else:
        summary = {
            "num_tasks": len(resolved_tasks),
            "overall":   {},
            "by_level":  {},
            "tasks": [
                {"task_id": t.task_id, "task_level": t.task_level}
                for t in resolved_tasks
            ],
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _write_init_stubs(resolved_tasks: List[ResolvedTask], per_task_root: Path) -> None:
    """
    Write skeleton judge_report.json and control_report.json for each task so
    human_eval_server.py can list and display tasks without any LLM judging.

    Only creates files that don't yet exist — safe to re-run.
    """
    for task in resolved_tasks:
        task_dir = per_task_root / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # judge_report.json — skeleton with questions from YAML, no LLM answers
        jr_path = task_dir / "judge_report.json"
        if not jr_path.exists():
            jr = {
                "task_id":    task.task_id,
                "init_frame": str(task.init_frame),
                "final_frame": str(task.final_frame) if task.final_frame.exists() else "",
                "answers": [
                    {"id": q.id, "answer": None, "confidence": None, "notes": ""}
                    for q in task.questions
                ],
                "score": {},
            }
            jr_path.write_text(json.dumps(jr, indent=2, ensure_ascii=False), encoding="utf-8")

        # control_report.json — skeleton with pre-computed request fields, no LLM judgments.
        # requested_trigger and requested_occlusion are derived from the task YAML using the
        # same logic as the real control judge, so the human eval UI displays them immediately.
        cr_path = task_dir / "control_report.json"
        if not cr_path.exists():
            try:
                video_wm, camera_wm, camera_pose = _load_task_fields(task.task_yaml)
                requested_trigger, requested_occlusion = _compute_requested_fields(
                    video_wm, camera_wm, camera_pose
                )
            except Exception:
                video_wm = ""
                requested_trigger = ""
                requested_occlusion = ""
            cr = {
                "task_id":             task.task_id,
                "video_WM_prompt":     video_wm,
                "wm_video":            str(task.wm_video),
                "requested_occlusion": requested_occlusion,
                "requested_trigger":   requested_trigger,
                "occlusion_done":      None,
                "trigger_applied":     None,
                "artifact":            None,
                "notes":               "",
            }
            cr_path.write_text(json.dumps(cr, indent=2, ensure_ascii=False), encoding="utf-8")


def _eval_one_task(
    task: ResolvedTask,
    *,
    provider: str,
    judge_model: str,
    control_model: str,
    run_judge: bool = True,
    run_control: bool = True,
) -> Tuple[Optional[JudgeResult], Optional[ControlJudgeResult]]:
    judge_result   = judge_one_task(task, provider=provider, model=judge_model) if run_judge   else None
    control_result = evaluate_control_one_task(task, model=control_model)       if run_control else None
    return judge_result, control_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", required=True, type=str, help="JSON mapping task_id -> video path or URL.")

    parser.add_argument(
        "--task_root",
        default="benchmark/tasks/",
        type=str,
        help="Root directory containing task folders."
    )
    parser.add_argument(
        "--run_dir",
        default="runs/",
        type=str,
        help="Directory to write normalized run artifacts."
    )

    # Judge LLM provider info
    parser.add_argument(
        "--judge_vlm_provider",
        default="gemini",
        type=str,
        help="Provider of the judge VLM, currently support OpenAI, Anthropic and Gemini. API key must be configured."
    )
    parser.add_argument(
        "--judge_vlm_model",
        default="gemini-3-pro-preview",
        type=str,
        help="Name of the judge VLM model."
    )
    parser.add_argument(
        "--control_judge_model",
        default="gemini-3-pro-preview",
        type=str,
        help="Name of the control judge model."
    )

    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        help="Number of parallel workers for per-task eval (judge + control judge).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        type=str,
        help="fnmatch pattern to filter task IDs (e.g. 'ice_on_burner*'). If omitted, all tasks are evaluated.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate tasks that already have both judge_report.json and control_report.json.",
    )
    parser.add_argument(
        "--download_urls",
        action="store_true",
        help="If set, download http(s) videos into the run directory.",
    )
    parser.add_argument(
        "--state_only",
        action="store_true",
        help="Run only the state evolution judge (judge_report). Skips control judge.",
    )
    parser.add_argument(
        "--control_only",
        action="store_true",
        help="Run only the control judge (control_report). Skips quality judge and scoring.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help=(
            "Initialize the run directory only: resolve tasks, create per-task folders, "
            "summary.json, and skeleton report stubs. Exits without running any LLM judge. "
            "Use this to prepare a run for human-only evaluation."
        ),
    )

    args = parser.parse_args()

    if args.state_only and args.control_only:
        parser.error("--state_only and --control_only are mutually exclusive.")

    run_judge   = not args.control_only
    run_control = not args.state_only

    # ---------------------------------------------------------------------------
    # Build eval run directory
    # ---------------------------------------------------------------------------
    task_root = Path(args.task_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()

    if not task_root.exists():
        raise FileNotFoundError(task_root)
    if not output_json.exists():
        raise FileNotFoundError(output_json)

    run_dir.mkdir(parents=True, exist_ok=True)

    run_name = output_json.stem
    run_root = run_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    per_task_root = run_root / "per_task"
    per_task_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Task resolution — sequential (final frame extraction happens here)
    # ---------------------------------------------------------------------------
    resolved_tasks: List[ResolvedTask] = resolve_tasks_from_outputs_json(
        tasks_root=task_root,
        outputs_json=output_json,
        run_dir=run_root,
        download_urls=args.download_urls,
        pattern=args.pattern,
    )

    print(f"[DONE] Resolved {len(resolved_tasks)} tasks into: {per_task_root}")

    # ---------------------------------------------------------------------------
    # Initialise summary.json skeleton (always, before any judging)
    # ---------------------------------------------------------------------------
    _init_summary_json(resolved_tasks, run_root)

    # ---------------------------------------------------------------------------
    # --init: write skeleton report stubs and exit — no LLM judging
    # ---------------------------------------------------------------------------
    if args.init:
        _write_init_stubs(resolved_tasks, per_task_root)
        print(f"[DONE] Initialized {len(resolved_tasks)} task(s) in: {run_root}")
        print(f"       summary.json, judge_report.json and control_report.json stubs written.")
        print(f"       Start human_eval_server.py to begin human evaluation.")
        return

    # Keep the full list for the summary-append step at the end.
    all_resolved_tasks = resolved_tasks

    # ---------------------------------------------------------------------------
    # Skip tasks that are already fully evaluated (unless --overwrite)
    # ---------------------------------------------------------------------------
    if not args.overwrite:
        pending, skipped = [], 0
        for task in resolved_tasks:
            task_dir = per_task_root / task.task_id
            judge_done   = (task_dir / "judge_report.json").exists()
            control_done = (task_dir / "control_report.json").exists()
            already_done = (
                (run_judge   and judge_done   or not run_judge) and
                (run_control and control_done or not run_control)
            )
            if already_done:
                skipped += 1
            else:
                pending.append(task)
        if skipped:
            print(f"[SKIP] {skipped} already-evaluated tasks (use --overwrite to re-run)")
        resolved_tasks = pending

    # ---------------------------------------------------------------------------
    # Per-task eval — parallel (judge + control judge run together per task)
    # ---------------------------------------------------------------------------
    judge_results: List[JudgeResult] = []
    control_results: List[ControlJudgeResult] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _eval_one_task,
                task,
                provider=args.judge_vlm_provider,
                judge_model=args.judge_vlm_model,
                control_model=args.control_judge_model,
                run_judge=run_judge,
                run_control=run_control,
            ): task
            for task in resolved_tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                jr, cr = future.result()
                if jr is not None:
                    judge_results.append(jr)
                if cr is not None:
                    control_results.append(cr)
            except Exception as e:
                print(f"[ERROR] {task.task_id}: {e}")
                failed += 1

    print(f"[DONE] Evaluated {len(judge_results)} tasks ({failed} failed)")

    # ---------------------------------------------------------------------------
    # Scoring — sequential (aggregates all judge results)
    # ---------------------------------------------------------------------------
    if run_judge and judge_results:
        task_scores = score_all_tasks(judge_results=judge_results)
        _ = write_run_report(task_scores=task_scores, run_dir=run_root)
        print(f"[DONE] Scored {len(task_scores)} tasks")

    # ---------------------------------------------------------------------------
    # Append control results to summary — sequential
    # Uses all_resolved_tasks so already-skipped tasks with existing
    # control_report.json are also merged into summary.json.
    # ---------------------------------------------------------------------------
    if run_control:
        append_control_results_to_summary(tasks=all_resolved_tasks)
        print(f"[DONE] Controllability evaluation done")

    return


if __name__ == "__main__":
    main()
