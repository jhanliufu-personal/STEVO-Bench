import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from eval.task_resolver import resolve_tasks_from_outputs_json, ResolvedTask
from eval.judge_runner import judge_one_task
from eval.score_task import score_all_tasks, write_run_report
from eval.control_judge import evaluate_control_one_task, append_control_results_to_summary
from eval.judge_output_parser import JudgeResult
from eval.control_judge import ControlJudgeResult


def _eval_one_task(
    task: ResolvedTask,
    *,
    provider: str,
    judge_model: str,
    control_model: str,
) -> Tuple[JudgeResult, ControlJudgeResult]:
    judge_result = judge_one_task(task, provider=provider, model=judge_model)
    control_result = evaluate_control_one_task(task, model=control_model)
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

    args = parser.parse_args()

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
    # Skip tasks that are already fully evaluated (unless --overwrite)
    # ---------------------------------------------------------------------------
    if not args.overwrite:
        pending, skipped = [], 0
        for task in resolved_tasks:
            task_dir = per_task_root / task.task_id
            if (task_dir / "judge_report.json").exists() and (task_dir / "control_report.json").exists():
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
            ): task
            for task in resolved_tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                jr, cr = future.result()
                judge_results.append(jr)
                control_results.append(cr)
            except Exception as e:
                print(f"[ERROR] {task.task_id}: {e}")
                failed += 1

    print(f"[DONE] Evaluated {len(judge_results)} tasks ({failed} failed)")

    # ---------------------------------------------------------------------------
    # Scoring — sequential (aggregates all judge results)
    # ---------------------------------------------------------------------------
    task_scores = score_all_tasks(judge_results=judge_results)
    _ = write_run_report(task_scores=task_scores, run_dir=run_root)
    print(f"[DONE] Scored {len(task_scores)} tasks")

    # ---------------------------------------------------------------------------
    # Append control results to summary — sequential
    # ---------------------------------------------------------------------------
    append_control_results_to_summary(tasks=resolved_tasks)
    print(f"[DONE] Controllability evaluation done")

    return


if __name__ == "__main__":
    main()
