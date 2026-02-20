#!/usr/bin/env python3
"""
Clean up task directories to regenerate from scratch:
1. Remove all .png files (init frames, final frames, etc.)
2. Remove evaluation sections from all YAML files

Usage:
  python benchmark/runners/cleanup_tasks.py --tasks_root benchmark/tasks --dry-run
  python benchmark/runners/cleanup_tasks.py --tasks_root benchmark/tasks/level1_scalar_state
  python benchmark/runners/cleanup_tasks.py --tasks_root benchmark/tasks --confirm
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from utils import load_yaml, dump_yaml


def find_png_files(root: Path) -> List[Path]:
    """Find all .png files in task directories."""
    png_files = []
    for png in root.rglob("*.png"):
        # Skip if in hidden directories or __pycache__
        if any(part.startswith('.') or part == '__pycache__' for part in png.parts):
            continue
        png_files.append(png)
    return png_files


def find_yaml_files(root: Path) -> List[Path]:
    """Find all .yaml files in task directories."""
    yaml_files = []
    for yaml in root.rglob("*.yaml"):
        # Skip if in hidden directories or __pycache__
        if any(part.startswith('.') or part == '__pycache__' for part in yaml.parts):
            continue
        yaml_files.append(yaml)
    return yaml_files


def remove_evaluation_from_yaml(yaml_path: Path, dry_run: bool = False) -> bool:
    """Remove evaluation section from a YAML file. Returns True if modified."""
    try:
        task = load_yaml(yaml_path)

        if "evaluation" in task:
            if dry_run:
                print(f"  [DRY RUN] Would remove evaluation from: {yaml_path.name}")
                return True
            else:
                del task["evaluation"]
                dump_yaml(yaml_path, task)
                print(f"  Removed evaluation from: {yaml_path.name}")
                return True

        return False
    except Exception as e:
        print(f"  [ERROR] Failed to process {yaml_path}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean up task directories for fresh regeneration"
    )
    parser.add_argument(
        "--tasks_root",
        type=str,
        required=True,
        help="Root directory containing task subdirectories (e.g., benchmark/tasks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt and proceed with cleanup",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Only remove PNG files, keep evaluation sections",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only remove evaluation sections, keep PNG files",
    )

    args = parser.parse_args()

    tasks_root = Path(args.tasks_root).resolve()

    if not tasks_root.exists():
        print(f"Error: Directory not found: {tasks_root}", file=sys.stderr)
        return 1

    # Find files to clean
    print("Scanning for files to clean...")
    png_files = [] if args.eval_only else find_png_files(tasks_root)
    yaml_files = [] if args.png_only else find_yaml_files(tasks_root)

    # Show summary
    print(f"\nFound in {tasks_root}:")
    print(f"  PNG files: {len(png_files)}")
    print(f"  YAML files: {len(yaml_files)}")

    if not png_files and not yaml_files:
        print("\nNothing to clean!")
        return 0

    # Show samples
    if png_files:
        print(f"\nSample PNG files (showing first 5):")
        for png in png_files[:5]:
            rel_path = png.relative_to(tasks_root)
            print(f"  - {rel_path}")
        if len(png_files) > 5:
            print(f"  ... and {len(png_files) - 5} more")

    if yaml_files:
        print(f"\nWill check {len(yaml_files)} YAML files for evaluation sections")

    # Confirm unless --confirm or --dry-run
    if not args.dry_run and not args.confirm:
        print("\n" + "="*60)
        print("WARNING: This will permanently delete files!")
        print("="*60)
        response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            print("Cleanup cancelled.")
            return 0

    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - No files will be modified")
        print("="*60)

    # Clean PNG files
    png_deleted = 0
    if png_files:
        print(f"\nRemoving PNG files...")
        for png in png_files:
            if args.dry_run:
                print(f"  [DRY RUN] Would delete: {png.relative_to(tasks_root)}")
                png_deleted += 1
            else:
                try:
                    png.unlink()
                    print(f"  Deleted: {png.relative_to(tasks_root)}")
                    png_deleted += 1
                except Exception as e:
                    print(f"  [ERROR] Failed to delete {png}: {e}")

    # Clean evaluation sections
    yaml_modified = 0
    if yaml_files:
        print(f"\nRemoving evaluation sections from YAML files...")
        for yaml_path in yaml_files:
            if remove_evaluation_from_yaml(yaml_path, args.dry_run):
                yaml_modified += 1

    # Summary
    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    if args.dry_run:
        print(f"[DRY RUN] Would delete: {png_deleted} PNG files")
        print(f"[DRY RUN] Would modify: {yaml_modified} YAML files")
    else:
        print(f"Deleted: {png_deleted} PNG files")
        print(f"Modified: {yaml_modified} YAML files (removed evaluation)")
    print("="*60)

    if args.dry_run:
        print("\nRun without --dry-run to actually perform cleanup")
    else:
        print("\nCleanup complete! Ready for fresh generation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
