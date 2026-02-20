#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# script_name = "generate_init_frame.py"
# script_name = "generate_gt_and_questions.py"
script_name = "generate_gt_final_frame.py"
SCRIPT_PATH = Path(
    rf"G:\My Drive\Gkioxari_Lab\StateWMBench\StateWM\benchmark\runners\{script_name}"
)

# Lock for thread-safe printing
print_lock = Lock()

def run_task(python_bin: str, task_dir: Path) -> tuple[Path, int]:
    """Run generator for one task dir. Returns (task_dir, returncode)."""
    cmd = [python_bin, str(SCRIPT_PATH), "--task_path", str(task_dir)]

    with print_lock:
        print(f"[START] Task: {task_dir}")

    # Capture output to avoid interleaved output from parallel processes
    result = subprocess.run(cmd, capture_output=True, text=True)

    with print_lock:
        if result.stdout:
            print(result.stdout, end='')
        if result.stderr:
            print(result.stderr, end='', file=sys.stderr)

        if result.returncode != 0:
            print(f"[FAIL] Task failed: {task_dir} (exit code {result.returncode})")
        else:
            print(f"[DONE] Task completed: {task_dir}")

    return task_dir, result.returncode


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 for sequential execution).",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root_dir).expanduser().resolve()
    python_bin = args.python_bin
    num_workers = args.workers

    if not tasks_root.is_dir():
        print(f"Error: TASKS_ROOT_DIR does not exist: {tasks_root}", file=sys.stderr)
        return 1

    if not SCRIPT_PATH.is_file():
        print(f"Error: Python script not found: {SCRIPT_PATH}", file=sys.stderr)
        return 1

    if num_workers < 1:
        print(f"Error: --workers must be >= 1, got {num_workers}", file=sys.stderr)
        return 1

    print("Running world-model benchmark GT generation")
    print(f"Tasks root: {tasks_root}")
    print(f"Python: {python_bin}")
    print(f"Workers: {num_workers}")
    print("--------------------------------------------")

    # Collect valid task directories
    valid_tasks = []
    for task_dir in iter_dirs(tasks_root):
        if not task_dir.is_dir():
            continue

        basename = task_dir.name
        yaml_path = task_dir / f"{basename}.yaml"

        if not yaml_path.is_file():
            print(f"[SKIP] {task_dir} (missing {basename}.yaml)")
            continue

        valid_tasks.append(task_dir)

    if not valid_tasks:
        print(f"No valid task directories found under: {tasks_root}", file=sys.stderr)
        return 1

    print(f"\nFound {len(valid_tasks)} valid task(s)")
    print()

    # Run tasks in parallel
    failed_tasks = []

    if num_workers == 1:
        # Sequential execution (original behavior, cleaner output)
        for task_dir in valid_tasks:
            _, status = run_task(python_bin, task_dir)
            if status != 0:
                failed_tasks.append(task_dir)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(run_task, python_bin, task_dir): task_dir
                for task_dir in valid_tasks
            }

            # Wait for completion and collect results
            for future in as_completed(futures):
                task_dir, status = future.result()
                if status != 0:
                    failed_tasks.append(task_dir)

    print()
    if failed_tasks:
        print(f"Completed with {len(failed_tasks)} failure(s):")
        for task_dir in failed_tasks:
            print(f"  - {task_dir}")
        return 1
    else:
        print(f"Successfully generated for all {len(valid_tasks)} task(s)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
