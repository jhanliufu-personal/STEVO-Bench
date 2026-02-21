#!/usr/bin/env python3
"""
generate_gt_final_frame.py

Given a task path (folder or YAML file), this script:
  1) Locates the task YAML and init frame using your naming convention:
       - <task_dir>/<task_dir_name>.yaml
       - <task_dir>/<task_dir_name>_init_frame.png
  2) Loads the image-edit prompt from YAML:
       evaluation.init2final_edit_prompt
  3) Sends (init_frame, edit_prompt) to Nano Banana image-edit API
  4) Saves the edited image into the same task directory as:
       <task_dir_name>_gt_final_frame.png

Environment variables:
    - GOOGLE_API_KEY

Usage (single task):
  python generate_gt_final_frame.py --task_path path/to/task_dir

Usage (pattern — runs for all matching task folders):
  python generate_gt_final_frame.py \
    --pattern "pouring_water_into_cup*" \
    --tasks_root benchmark/tasks/

Optional:
  --overwrite    overwrite output file if exists
"""

import os
import argparse
import fnmatch
from pathlib import Path
from typing import List

from utils import resolve_task_paths, load_yaml, call_nanobanana


# -----------------------------
# Pattern helper
# -----------------------------

def _find_pattern_tasks(pattern: str, tasks_root: str) -> List[Path]:
    """Return task dirs under tasks_root whose folder name matches pattern."""
    root = Path(tasks_root).expanduser().resolve()
    results = []
    for d in sorted(root.rglob("*")):
        if d.is_dir() and fnmatch.fnmatch(d.name, pattern):
            if (d / f"{d.name}.yaml").exists():
                results.append(d)
    return results


# -----------------------------
# Per-task runner
# -----------------------------

def _run_single(task_path: Path, model: str, overwrite: bool, api_key: str) -> bool:
    """Generate the GT final frame for a single task. Returns True on success."""
    task_dir, task_yaml_path, folder_name, init_frame_path = resolve_task_paths(task_path)
    task = load_yaml(task_yaml_path)
    eval_block = (task.get("evaluation") or {})

    final_frame_path = (task_dir / f"{folder_name}_gt_final_frame.png").resolve()
    if final_frame_path.exists() and not overwrite:
        print(f"GT final frame already exists: {final_frame_path.name} (use --overwrite to regenerate)")
        return True

    # Load edit prompt (init2final)
    edit_prompt = (eval_block.get("init2final_edit_prompt") or "").strip()
    if not edit_prompt:
        raise ValueError(
            f"Missing evaluation.init2final_edit_prompt in YAML: {task_yaml_path}"
        )

    result = call_nanobanana(
        api_key=api_key,
        model=model,
        input_image=init_frame_path,
        output_path=final_frame_path,
        prompt=edit_prompt,
        # overwrite=overwrite
    )

    if result and final_frame_path.exists():
        print(f"[OK] Generated GT final frame: {final_frame_path.name}")
        return True
    else:
        print(f"[FAIL] No image returned by model — GT final frame was NOT saved: {final_frame_path.name}")
        return False


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task_path",
        type=str,
        help="Task folder path OR task YAML path (single task).",
    )
    group.add_argument(
        "--pattern",
        type=str,
        help="Glob pattern matching task folder names, e.g. 'pouring_water_into_cup*'.",
    )
    parser.add_argument(
        "--tasks_root",
        default="benchmark/tasks/",
        type=str,
        help="Root directory to search when --pattern is used (default: benchmark/tasks/).",
    )
    parser.add_argument("--model", default="gemini-3-pro-image-preview", type=str)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output image if it already exists.")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

    if args.task_path:
        ok = _run_single(Path(args.task_path), args.model, args.overwrite, api_key)
        if not ok:
            raise SystemExit(1)
    else:
        tasks = _find_pattern_tasks(args.pattern, args.tasks_root)
        if not tasks:
            print(f"No tasks found matching pattern '{args.pattern}' under {args.tasks_root}")
            raise SystemExit(1)
        print(f"Found {len(tasks)} task(s) matching '{args.pattern}'")
        failed = []
        for task_dir in tasks:
            print(f"\n--- {task_dir.name} ---")
            try:
                ok = _run_single(task_dir, args.model, args.overwrite, api_key)
            except Exception as e:
                print(f"[ERROR] {task_dir.name}: {e}")
                ok = False
            if not ok:
                failed.append(task_dir.name)
        print(f"\n{'=' * 40}")
        print(f"Done: {len(tasks) - len(failed)}/{len(tasks)} succeeded.")
        if failed:
            print(f"Failed ({len(failed)}): {', '.join(failed)}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
