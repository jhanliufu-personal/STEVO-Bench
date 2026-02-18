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

Usage:
  python generate_gt_final_frame.py --task_path path/to/task_dir

Optional:
  --out_name <filename.png>   (default: <folder>_gt_final_frame.png)
  --overwrite                 overwrite output file if exists
"""

import os
import argparse
from pathlib import Path

from utils import resolve_task_paths, load_yaml, call_nanobanana

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_path", required=True, type=str, help="Task folder path OR task YAML path.")
    parser.add_argument("--model", default="gemini-3-pro-image-preview", type=str)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output image if it already exists.")
    args = parser.parse_args()

    # Load task spec
    task_dir, task_yaml_path, folder_name, init_frame_path = resolve_task_paths(Path(args.task_path))
    task = load_yaml(task_yaml_path)
    eval_block = (task.get("evaluation") or {})

    final_frame_path = (task_dir / f"{folder_name}_gt_final_frame.png").resolve()
    if final_frame_path.exists() and not args.overwrite:
        print("GT final frame already exists and --overwrite is not set. Returning.")
        return

    # Load edit prompt (init2final)
    edit_prompt = (eval_block.get("init2final_edit_prompt") or "").strip()
    if not edit_prompt:
        raise ValueError(
            f"Missing evaluation.init2final_edit_prompt in YAML: {task_yaml_path}"
        )
        
    # Call Nanobanana
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

    final_frame_path = (task_dir / f"{folder_name}_gt_final_frame.png").resolve()
    _ = call_nanobanana(
        api_key=api_key,
        model=args.model,
        input_image=init_frame_path,
        output_path=final_frame_path,
        prompt=edit_prompt,
        # overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()