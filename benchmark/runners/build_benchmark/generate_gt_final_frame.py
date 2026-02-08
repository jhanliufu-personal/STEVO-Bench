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
  OR
  python generate_gt_final_frame.py --task_path path/to/task_dir/task_dir.yaml

Optional:
  --out_name <filename.png>   (default: <folder>_gt_final_frame.png)
  --overwrite                 overwrite output file if exists
"""

import os
import argparse
from pathlib import Path

from utils import resolve_task_paths
from utils import load_yaml

from google import genai
# from google.genai import types
from PIL import Image

# -----------------------------
# Nano Banana
# -----------------------------
def call_nanobanana(
    api_key: str,
    model: str,
    initial_frame_path: Path,
    final_frame_path: Path,
    edit_prompt: str,
    overwrite: bool = False
) -> Path:
    # Validate inputs
    if not api_key or not api_key.strip():
        raise ValueError("api_key is empty. Provide a valid Google API key.")
    if not model or not model.strip():
        raise ValueError("model is empty.")
    if not edit_prompt or not edit_prompt.strip():
        raise ValueError("edit_prompt is empty.")

    initial_frame_path = Path(initial_frame_path).expanduser().resolve()
    final_frame_path = Path(final_frame_path).expanduser().resolve()

    if not initial_frame_path.exists():
        raise FileNotFoundError(f"Initial frame does not exist: {initial_frame_path}")
    if not initial_frame_path.is_file():
        raise ValueError(f"Initial frame path is not a file: {initial_frame_path}")

    if final_frame_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {final_frame_path}. Use overwrite=True to replace it."
        )

    # Load initial frame
    image = Image.open(initial_frame_path)

    client = genai.Client(api_key=api_key)

    
    response = client.models.generate_content(
        model=model,
        contents=[edit_prompt, image],
    )

    if not response.parts:
        raise RuntimeError(
            "Gemini response contained no parts. "
        )

    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = part.as_image()
            image.save(final_frame_path)

    return final_frame_path

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
        initial_frame_path=init_frame_path,
        final_frame_path=final_frame_path,
        edit_prompt=edit_prompt,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()