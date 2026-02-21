#!/usr/bin/env python3
"""
Generate initial frame for a task.

Handles three situations:
1. No init frame exists -> Generate from scratch using image_gen_prompt
2. Init frame exists but from different task (variant) -> Edit using init_frame_edit_prompt
3. Init frame exists and matches current task -> Skip unless --overwrite

Usage (single task):
  python benchmark/runners/generate_init_frame.py \
    --task_path benchmark/tasks/level1_scalar_state/ice_on_burner_01 \
    --model imagen-3.0-generate-001 \
    --overwrite

Usage (pattern — runs for all matching task folders):
  python benchmark/runners/generate_init_frame.py \
    --pattern "pouring_water_into_cup*" \
    --tasks_root benchmark/tasks/ \
    --model imagen-3.0-generate-001
"""

import os
import argparse
import fnmatch
from pathlib import Path
from typing import List, Optional, Tuple

from utils import resolve_task_paths, load_yaml, call_nanobanana


def find_existing_init_frame(task_dir: Path) -> Optional[Tuple[Path, str]]:
    """
    Find existing *_init_frame.png in task directory.

    Returns:
        (frame_path, base_name) if found, None otherwise
        base_name is the ** part of **_init_frame.png
    """
    pattern = "*_init_frame.png"
    matches = list(task_dir.glob(pattern))

    if not matches:
        return None

    if len(matches) > 1:
        print(f"Warning: Multiple init frames found in {task_dir}, using first: {matches[0].name}")

    frame_path = matches[0]
    # Extract base name: "task_01_init_frame.png" -> "task_01"
    base_name = frame_path.stem.replace("_init_frame", "")

    return frame_path, base_name


def _find_pattern_tasks(pattern: str, tasks_root: str) -> List[Path]:
    """Return task dirs under tasks_root whose folder name matches pattern."""
    root = Path(tasks_root).expanduser().resolve()
    results = []
    for d in sorted(root.rglob("*")):
        if d.is_dir() and fnmatch.fnmatch(d.name, pattern):
            if (d / f"{d.name}.yaml").exists():
                results.append(d)
    return results


def _run_single(task_path: Path, model: str, overwrite: bool) -> bool:
    """Run init frame generation for a single task. Returns True on success."""
    task_dir, task_yaml_path, folder_name, _ = resolve_task_paths(task_path)
    task = load_yaml(task_yaml_path)

    task_id = task.get("id", folder_name)
    prompts_block = task.get("prompts") or {}

    # Check for existing init frame
    existing_frame = find_existing_init_frame(task_dir)
    output_path = task_dir / f"{task_id}_init_frame.png"

    # Determine situation
    if existing_frame:
        existing_path, existing_base_name = existing_frame

        # Situation 2: Frame from different task (variant)
        if existing_base_name != task_id:
            print(f"Found init frame from different task: {existing_base_name}")
            print(f"-> Will edit existing frame for variant: {task_id}")

            # Get edit prompt
            init_frame_edit = prompts_block.get("init_frame_edit_prompt", "").strip()

            if not init_frame_edit:
                raise ValueError(
                    f"Variant task missing prompts.init_frame_edit_prompt in YAML: {task_yaml_path}"
                )

            if init_frame_edit.upper() == "[NO CHANGE]":
                print("-> Edit prompt is [NO CHANGE]")

                # Rename existing frame to match current task if needed
                if existing_path != output_path:
                    print(f"-> Renaming {existing_path.name} to {output_path.name}")
                    existing_path.rename(output_path)
                else:
                    print("-> Frame already has correct name, keeping as-is")

                print("[OK] Init frame ready (no changes)")
                return True

            # Call image editing model
            api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

            print(f"-> Editing with prompt: {init_frame_edit[:80]}...")

            result = call_nanobanana(
                api_key=api_key,
                model=model,
                prompt=init_frame_edit,
                input_image=existing_path,
                output_path=output_path,
                overwrite=True,  # Always generate for variants
            )

            if result and output_path.exists():
                print(f"[OK] Generated edited init frame: {output_path.name}")
                # Clean up old frame if different
                if existing_path != output_path:
                    existing_path.unlink()
                    print(f"-> Removed old frame: {existing_path.name}")
                return True
            else:
                print(f"[FAIL] No image returned by model — init frame was NOT saved: {output_path.name}")
                return False

        # Situation 3: Frame already exists for this task
        else:
            if not overwrite:
                print(f"Init frame already exists: {existing_path.name}")
                print("-> Use --overwrite to regenerate")
                return True

            print(f"Init frame exists, but --overwrite is set")
            print(f"-> Regenerating init frame for: {task_id}")

            # Fall through to Situation 1 logic below

    # Situation 1: Generate from scratch
    if not existing_frame or (existing_frame and overwrite):
        init_gen_prompt = prompts_block.get("image_gen_prompt", "").strip()

        if not init_gen_prompt:
            raise ValueError(
                f"Missing prompts.image_gen_prompt in YAML: {task_yaml_path}"
            )

        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

        print(f"-> Generating from scratch: {task_id}")
        print(f"-> Prompt: {init_gen_prompt[:80]}...")

        result = call_nanobanana(
            api_key=api_key,
            model=model,
            prompt=init_gen_prompt,
            output_path=output_path,
            overwrite=overwrite,
        )

        if result and output_path.exists():
            print(f"[OK] Generated init frame: {output_path.name}")
            return True
        else:
            print(f"[FAIL] No image returned by model — init frame was NOT saved: {output_path.name}")
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate initial frame for a task (handles variants)"
    )
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
    parser.add_argument(
        "--model",
        default="gemini-3-pro-image-preview",
        type=str,
        help="Image generation model",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output image if it already exists.",
    )
    args = parser.parse_args()

    if args.task_path:
        ok = _run_single(Path(args.task_path), args.model, args.overwrite)
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
                ok = _run_single(task_dir, args.model, args.overwrite)
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
