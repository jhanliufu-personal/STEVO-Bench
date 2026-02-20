#!/usr/bin/env python3
"""
Generate initial frame for a task.

Handles three situations:
1. No init frame exists -> Generate from scratch using image_gen_prompt
2. Init frame exists but from different task (variant) -> Edit using init_frame_edit_prompt
3. Init frame exists and matches current task -> Skip unless --overwrite

Usage:
  python benchmark/runners/generate_init_frame.py \
    --task_path benchmark/tasks/level1_scalar_state/ice_on_burner_01 \
    --model imagen-3.0-generate-001 \
    --overwrite
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate initial frame for a task (handles variants)"
    )
    parser.add_argument(
        "--task_path",
        required=True,
        type=str,
        help="Task folder path OR task YAML path."
    )
    parser.add_argument(
        "--model",
        default="gemini-3-pro-image-preview",
        type=str,
        help="Image generation model"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output image if it already exists."
    )
    args = parser.parse_args()

    # Load task spec
    task_dir, task_yaml_path, folder_name, _ = resolve_task_paths(Path(args.task_path))
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

                print("✓ Init frame ready (no changes)")
                return

            # Call image editing model
            api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

            print(f"-> Editing with prompt: {init_frame_edit[:80]}...")

            result = call_nanobanana(
                api_key=api_key,
                model=args.model,
                prompt=init_frame_edit,
                input_image=existing_path,
                output_path=output_path,
                overwrite=True,  # Always generate for variants
            )

            if result:
                print(f"✓ Generated edited init frame: {output_path.name}")
                # Clean up old frame if different
                if existing_path != output_path:
                    existing_path.unlink()
                    print(f"-> Removed old frame: {existing_path.name}")

        # Situation 3: Frame already exists for this task
        else:
            if not args.overwrite:
                print(f"Init frame already exists: {existing_path.name}")
                print("-> Use --overwrite to regenerate")
                return

            print(f"Init frame exists, but --overwrite is set")
            print(f"-> Regenerating init frame for: {task_id}")

            # Fall through to Situation 1 logic below

    # Situation 1: Generate from scratch
    if not existing_frame or (existing_frame and args.overwrite):
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
            model=args.model,
            prompt=init_gen_prompt,
            output_path=output_path,
            overwrite=args.overwrite,
        )

        if result:
            print(f" Generated init frame: {output_path.name}")


if __name__ == "__main__":
    main()