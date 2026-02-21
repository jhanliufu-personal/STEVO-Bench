#!/usr/bin/env python3
"""Generate world-model videos for all benchmark tasks.

Usage:
    python -m generation.run_world_models \\
        --models veo hunyuan \\
        --tasks_root benchmark/tasks/ \\
        --output_root outputs/ \\
        --run_name test_2 \\
        --workers 4

    # Run all configured models:
    python -m generation.run_world_models --models all --run_name test_2

    # Re-generate even if output already exists:
    python -m generation.run_world_models --models veo --run_name test_2 --overwrite

Output layout (mirrors existing convention):
    outputs/{model}_{run_name}/
        {task_id}.mp4
        {model}_output_map_{run_name}.json   ← written incrementally after each success
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import yaml

from generation.world_models import WorldModelRunner, build_runner


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def discover_tasks(tasks_root: Path) -> List[Tuple[str, dict, Optional[Path]]]:
    """Walk tasks_root recursively and return one entry per valid task directory.

    A directory is treated as a task dir if it contains a YAML file named
    after the directory itself (e.g. tasks/ice_on_burner_01/ice_on_burner_01.yaml).

    Returns:
        List of (task_id, task_yaml_data, init_frame_path_or_None).
        task_id is taken from the YAML 'id' field, falling back to the folder name.
    """
    tasks_root = Path(tasks_root).expanduser().resolve()
    found: List[Tuple[str, dict, Optional[Path]]] = []

    for d in sorted(tasks_root.rglob("*")):
        if not d.is_dir():
            continue
        yaml_path = d / f"{d.name}.yaml"
        if not yaml_path.exists():
            continue

        with yaml_path.open(encoding="utf-8") as f:
            task = yaml.safe_load(f) or {}

        task_id = str(task.get("id") or d.name)
        init_frame = d / f"{d.name}_init_frame.png"
        found.append((task_id, task, init_frame if init_frame.exists() else None))

    return found


# ---------------------------------------------------------------------------
# Output map helpers — written incrementally, thread-safe
# ---------------------------------------------------------------------------

def _update_map(map_path: Path, task_id: str, filename: str, lock: Lock) -> None:
    """Atomically add/update one entry in the JSON output map."""
    with lock:
        data: dict = {}
        if map_path.exists():
            data = json.loads(map_path.read_text(encoding="utf-8"))
        data[task_id] = filename
        map_path.write_text(json.dumps(data, indent=4), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-task work unit
# ---------------------------------------------------------------------------

def _run_one(
    runner: WorldModelRunner,
    task_id: str,
    task: dict,
    init_frame: Optional[Path],
    output_path: Path,
) -> bool:
    """Extract the right prompt + camera_control from the task YAML and call generate()."""
    prompts = task.get("prompts") or {}
    prompt = (prompts.get(runner.prompt_field) or "").strip()
    if not prompt:
        print(
            f"[{runner.name}] SKIP {task_id}: "
            f"prompt field '{runner.prompt_field}' is empty or missing"
        )
        return False

    camera_control: Optional[str] = None
    if runner.camera_control_field:
        cc = (task.get("camera_control") or {}).get(runner.camera_control_field)
        if cc and str(cc).strip().lower() not in ("null", "none", ""):
            camera_control = str(cc).strip()

    return runner.generate(
        task_id=task_id,
        prompt=prompt,
        init_frame=init_frame,
        output_path=output_path,
        camera_control=camera_control,
    )


# ---------------------------------------------------------------------------
# Per-model run loop
# ---------------------------------------------------------------------------

def run_model(
    *,
    runner: WorldModelRunner,
    model_type: str,
    tasks: List[Tuple[str, dict, Optional[Path]]],
    out_dir: Path,
    run_name: str,
    workers: int,
    overwrite: bool,
) -> Dict[str, bool]:
    """Run all tasks through one model. Returns {task_id: success}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / f"{runner.name}_output_map_{run_name}.json"
    map_lock = Lock()

    # Build work queue — skip tasks that are already done unless --overwrite
    work_queue: List[Tuple[str, dict, Optional[Path], Path]] = []
    for task_id, task, init_frame in tasks:
        out_path = out_dir / f"{task_id}.mp4"
        if out_path.exists() and not overwrite:
            print(f"[{runner.name}] SKIP {task_id} (already exists)")
            continue
        work_queue.append((task_id, task, init_frame, out_path))

    print(f"\n[{runner.name}] {len(work_queue)} task(s) to generate → {out_dir}")

    results: Dict[str, bool] = {}

    def do_one(item: Tuple) -> Tuple[str, bool]:
        task_id, task, init_frame, out_path = item
        ok = _run_one(runner, task_id, task, init_frame, out_path)
        if ok:
            _update_map(map_path, task_id, out_path.name, map_lock)
            print(f"[{runner.name}] DONE  {task_id}")
        else:
            print(f"[{runner.name}] FAIL  {task_id}")
        return task_id, ok

    # Local models run sequentially to avoid GPU resource contention.
    # API models can be parallelized via --workers.
    effective_workers = 1 if model_type == "local" else max(1, workers)

    if effective_workers == 1:
        for item in work_queue:
            task_id, ok = do_one(item)
            results[task_id] = ok
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(do_one, item): item[0] for item in work_queue}
            for future in as_completed(futures):
                task_id, ok = future.result()
                results[task_id] = ok

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_models_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("models", {})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark tasks through world models and collect video outputs."
    )
    parser.add_argument(
        "--models", nargs="+", required=True, metavar="MODEL",
        help="Model name(s) defined in configs/models.yaml, or 'all' to run every model.",
    )
    parser.add_argument(
        "--tasks_root", default="benchmark/tasks/",
        help="Root directory of task YAML folders (default: benchmark/tasks/).",
    )
    parser.add_argument(
        "--output_root", default="outputs/",
        help="Root directory for model output folders (default: outputs/).",
    )
    parser.add_argument(
        "--run_name", required=True,
        help="Label appended to each output folder, e.g. 'test_2' → outputs/veo_test_2/.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel workers for API models (default: 1). "
            "Local models always run sequentially regardless of this setting."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate videos that already exist on disk.",
    )
    parser.add_argument(
        "--config", default="generation/configs/models.yaml",
        help="Path to the model registry YAML (default: generation/configs/models.yaml).",
    )
    args = parser.parse_args()

    # ---- Load config & instantiate runners ----
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        return 1

    models_config = _load_models_config(config_path)
    requested = list(models_config) if args.models == ["all"] else args.models

    runners: Dict[str, WorldModelRunner] = {}
    for name in requested:
        if name not in models_config:
            print(
                f"[ERROR] Unknown model '{name}'. "
                f"Available: {list(models_config)}",
                file=sys.stderr,
            )
            return 1
        try:
            runners[name] = build_runner(name, models_config[name])
        except Exception as e:
            print(f"[ERROR] Could not initialize runner for '{name}': {e}", file=sys.stderr)
            return 1

    # ---- Discover tasks ----
    tasks_root = Path(args.tasks_root).expanduser().resolve()
    if not tasks_root.exists():
        print(f"[ERROR] tasks_root not found: {tasks_root}", file=sys.stderr)
        return 1

    tasks = discover_tasks(tasks_root)
    if not tasks:
        print(f"[ERROR] No tasks found under: {tasks_root}", file=sys.stderr)
        return 1

    print(f"Discovered {len(tasks)} task(s) under {tasks_root}")

    output_root = Path(args.output_root).expanduser().resolve()

    # ---- Run each model in sequence (tasks within a model may be parallel) ----
    overall_failed: Dict[str, List[str]] = {}

    for model_name, runner in runners.items():
        out_dir = output_root / f"{model_name}_{args.run_name}"
        results = run_model(
            runner=runner,
            model_type=models_config[model_name].get("type", "local"),
            tasks=tasks,
            out_dir=out_dir,
            run_name=args.run_name,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        failed = [tid for tid, ok in results.items() if not ok]
        if failed:
            overall_failed[model_name] = failed

    # ---- Summary ----
    print("\n" + "=" * 50)
    if not overall_failed:
        print(f"All tasks completed successfully across {len(runners)} model(s).")
        return 0
    else:
        for model_name, failed_tasks in overall_failed.items():
            print(f"[{model_name}] {len(failed_tasks)} failure(s):")
            for t in failed_tasks:
                print(f"    - {t}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
