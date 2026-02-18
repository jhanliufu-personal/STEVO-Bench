import os
import argparse
from pathlib import Path

from utils import resolve_task_paths, load_yaml, call_nanobanana

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_path", required=True, type=str, help="Task folder path OR task YAML path.")
    parser.add_argument("--model", default="gemini-3-pro-image-preview", type=str)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output image if it already exists.")
    args = parser.parse_args()

    # Load task spec
    task_dir, task_yaml_path, folder_name, _ = resolve_task_paths(Path(args.task_path))
    task = load_yaml(task_yaml_path)

    promtps_block = (task.get("prompts") or {})
    init_gen_prompt = (promtps_block.get("image_gen_prompt") or "").strip()
    if not init_gen_prompt:
        raise ValueError(
            f"Missing prompts.init_gen_prompt in YAML: {task_yaml_path}"
        )
        
    # Call Nanobanana
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")
    
    init_frame_path = (task_dir / f"{folder_name}_init_frame.png").resolve()
    _ = call_nanobanana(
        api_key=api_key,
        model=args.model,
        output_path=init_frame_path,
        prompt=init_gen_prompt,
        # overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()