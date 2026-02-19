#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


script_name = "generate_init_frame.py"
# script_name = "generate_gt_and_questions.py"
# script_name = "generate_gt_final_frame.py"
SCRIPT_PATH = Path(
    rf"G:\My Drive\Gkioxari_Lab\StateWMBench\StateWM\benchmark\runners\{script_name}"
)

def run_task(python_bin: str, task_dir: Path) -> int:
    """Run generator for one task dir. Returns process returncode."""
    cmd = [python_bin, str(SCRIPT_PATH), "--task_path", str(task_dir)]
    # Mirror bash behavior: show child output directly, don't capture
    return subprocess.call(cmd)


def iter_dirs(root: Path):
    """Yield all directories under root, including root itself (like find -type d)."""
    # Include root itself to match `find root -type d`
    yield root
    for dirpath, dirnames, _filenames in os.walk(root):
        for d in dirnames:
            yield Path(dirpath) / d


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-generate world-model benchmark GT final frames and questions."
    )
    parser.add_argument("--tasks_root_dir", help="Root directory containing task subdirectories.")
    parser.add_argument(
        "--python_bin",
        nargs="?",
        default="python",
        help="Python executable to use (default: python).",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root_dir).expanduser().resolve()
    python_bin = args.python_bin

    if not tasks_root.is_dir():
        print(f"Error: TASKS_ROOT_DIR does not exist: {tasks_root}", file=sys.stderr)
        return 1

    if not SCRIPT_PATH.is_file():
        print(f"Error: Python script not found: {SCRIPT_PATH}", file=sys.stderr)
        return 1

    print("Running world-model benchmark GT generation")
    print(f"Tasks root: {tasks_root}")
    print(f"Python: {python_bin}")
    print("--------------------------------------------")

    found_any = False

    for task_dir in iter_dirs(tasks_root):
        if not task_dir.is_dir():
            continue

        basename = task_dir.name
        yaml_path = task_dir / f"{basename}.yaml"
        # init_frame = task_dir / f"{basename}_init_frame.png"

        # Check naming convention
        if not yaml_path.is_file():
            print(f"[SKIP] {task_dir} (missing {basename}.yaml)")
            continue
        # if not init_frame.is_file():
        #     print(f"[SKIP] {task_dir} (missing {basename}_init_frame.png)")
        #     continue

        found_any = True
        print()
        print(f"[RUN ] Task: {task_dir}")

        status = run_task(python_bin, task_dir)

        if status != 0:
            print(f"[FAIL] Task failed: {task_dir} (exit code {status})")
        else:
            print(f"[DONE] Task completed: {task_dir}")

    if not found_any:
        print(f"No valid task directories found under: {tasks_root}", file=sys.stderr)
        return 1

    print()
    print("Generated GT final frame and questions for all tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
