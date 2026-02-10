import argparse
import json
from pathlib import Path
from typing import List

from eval.task_resolver import resolve_tasks_from_outputs_json, ResolvedTask
from eval.judge_runner import judge_all_tasks
from eval.score_task import score_all_tasks, write_run_report

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
        help="Name of the Judge VLM model"
    )

    # Optional knobs (leave minimal)
    parser.add_argument(
        "--link_mode",
        default="symlink",
        choices=["symlink", "copy"],
        help="Whether to symlink or copy videos/init frames into the run directory.",
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

    # Create per_task root
    per_task_root = run_root / "per_task"
    per_task_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Call task_resolver: it should create per_task/<task_id>/ and fill artifacts
    # ---------------------------------------------------------------------------
    resolved_tasks: List[ResolvedTask] = resolve_tasks_from_outputs_json(
        tasks_root=task_root,
        outputs_json=output_json,
        run_dir=run_root,
        link_mode=args.link_mode,
        download_urls=args.download_urls,
    )

    print(f"[DONE] Prepared run directory: {run_root}")
    print(f"[DONE] Resolved {len(resolved_tasks)} tasks into: {per_task_root}")

    # ---------------------------------------------------------------------------
    # Call judge runner
    # ---------------------------------------------------------------------------
    judge_results = judge_all_tasks(resolved_tasks, provider=args.judge_vlm_provider, model=args.judge_vlm_model)

    print(f"[DONE] Produced task reports for {len(judge_results)} tasks")

    # ---------------------------------------------------------------------------
    # Call scorer
    # ---------------------------------------------------------------------------
    task_scores = score_all_tasks(judge_results=judge_results)
    _ = write_run_report(task_scores=task_scores, run_dir=run_dir)

    return


if __name__ == "__main__":
    main()