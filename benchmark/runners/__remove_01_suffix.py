#!/usr/bin/env python3
"""
Remove _01 suffix from task folders and files.

Usage:
  python benchmark/runners/remove_01_suffix.py \
    --level_dir benchmark/tasks/level1_scalar_state \
    --dry_run

  # Remove --dry_run to actually rename
"""

import argparse
from pathlib import Path

from utils import load_yaml, dump_yaml


def process_task_directory(task_dir: Path, dry_run: bool = True) -> None:
    """Rename a task directory and its contents to remove _01 suffix."""

    folder_name = task_dir.name

    # Only process directories ending in _01
    if not folder_name.endswith("_01"):
        return

    new_folder_name = folder_name[:-3]  # Remove "_01"
    new_task_dir = task_dir.parent / new_folder_name

    print(f"\n{folder_name} → {new_folder_name}")

    # Find all files in the directory
    files = list(task_dir.glob("*"))

    for file_path in files:
        if file_path.is_file():
            old_name = file_path.name

            # Replace _01 in filename
            if "_01" in old_name:
                new_name = old_name.replace("_01", "")
                print(f"  {old_name} → {new_name}")

                # Special handling for YAML: update id field
                if old_name.endswith(".yaml"):
                    if not dry_run:
                        # Load, update id, save with new name
                        task = load_yaml(file_path)
                        if "id" in task and task["id"].endswith("_01"):
                            task["id"] = task["id"][:-3]

                        # Save to new location (will be in renamed dir)
                        temp_yaml = task_dir / new_name
                        dump_yaml(temp_yaml, task)
                        file_path.unlink()  # Remove old file
                        print(f"    → Updated id field in YAML")
                else:
                    # Regular file rename
                    if not dry_run:
                        new_path = task_dir / new_name
                        file_path.rename(new_path)

    # Rename the directory itself
    if not dry_run:
        task_dir.rename(new_task_dir)
        print(f"  ✓ Directory renamed")
    else:
        print(f"  [DRY RUN] Would rename directory")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove _01 suffix from task folders and files"
    )
    parser.add_argument(
        "--level_dir",
        type=str,
        required=True,
        help="Level directory (e.g., benchmark/tasks/level1_scalar_state)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be renamed without actually doing it"
    )

    args = parser.parse_args()

    level_dir = Path(args.level_dir).resolve()

    if not level_dir.exists() or not level_dir.is_dir():
        raise FileNotFoundError(f"Level directory not found: {level_dir}")

    # Find all task directories
    task_dirs = sorted([d for d in level_dir.iterdir() if d.is_dir()])

    print(f"{'='*60}")
    print(f"Processing {level_dir.name}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'='*60}")

    renamed_count = 0
    for task_dir in task_dirs:
        if task_dir.name.endswith("_01"):
            process_task_directory(task_dir, args.dry_run)
            renamed_count += 1

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"Would rename {renamed_count} task directories")
        print("Run without --dry_run to apply changes")
    else:
        print(f"✓ Renamed {renamed_count} task directories")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
