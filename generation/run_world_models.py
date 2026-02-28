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
import collections
import fnmatch
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from filelock import FileLock

import yaml

from generation.world_models import WorldModelRunner, build_runner

# ---------------------------------------------------------------------------
# Occlusion-variant deduplication (camera-controlled models only)
# ---------------------------------------------------------------------------

_OCCLUSION_SUFFIXES = ("_cardboard", "_curtain", "_lightoff")


def _occlusion_base(task_id: str) -> str:
    for s in _OCCLUSION_SUFFIXES:
        if task_id.endswith(s):
            return task_id[: -len(s)]
    return task_id


def _dedup_for_camera_model(
    tasks: List[Tuple[str, dict, Optional[Path]]],
) -> List[Tuple[str, dict, Optional[Path]]]:
    """For camera-controlled models, collapse occlusion variants to one per base name.

    Keeps the first variant encountered (tasks are sorted alphabetically by
    discover_tasks, so _cardboard wins over _curtain/_lightoff).
    The returned task_id is the base name (suffix stripped), so the output
    video and map entry use e.g. "ice_on_burner" rather than "ice_on_burner_cardboard".
    """
    seen: set = set()
    result: List[Tuple[str, dict, Optional[Path]]] = []
    for task_id, task, init_frame in tasks:
        base = _occlusion_base(task_id)
        if base not in seen:
            seen.add(base)
            result.append((base, task, init_frame))
    return result


# ---------------------------------------------------------------------------
# RPM limiter
# ---------------------------------------------------------------------------

class RpmLimiter:
    """Sliding-window rate limiter. Thread-safe.

    Tracks the start time of every request in the last 60 seconds.
    acquire() blocks until the number of requests in that window is
    below rpm, then stamps the current time and returns.
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self._timestamps: collections.deque = collections.deque()
        self._lock = Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                # Evict timestamps older than 60 s
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return
                # Sleep until the oldest slot expires
                wait = 60.0 - (now - self._timestamps[0])
            time.sleep(max(wait, 0.1))


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
        init_frame = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = d / f"{d.name}_init_frame{ext}"
            if candidate.exists():
                init_frame = candidate
                break
        found.append((task_id, task, init_frame))

    return found


# ---------------------------------------------------------------------------
# Output map helpers — written incrementally, thread-safe
# ---------------------------------------------------------------------------

def _update_map(map_path: Path, task_id: str, filename: str, lock: Lock) -> None:
    """Atomically add/update one entry in the JSON output map.

    Uses both a threading.Lock (intra-process) and a FileLock on disk
    (cross-process), so concurrent runs against the same output folder
    from separate terminals are safe.
    """
    lock_path = map_path.with_suffix(".lock")
    with lock:                          # serialise threads within this process
        with FileLock(str(lock_path)):  # serialise across processes
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
    rpm_limit: Optional[int],
) -> Dict[str, bool]:
    """Run all tasks through one model. Returns {task_id: success}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / f"{runner.name}_output_map_{run_name}.json"
    map_lock = Lock()
    limiter = RpmLimiter(rpm_limit) if rpm_limit else None

    # Build work queue — skip tasks that are already done unless --overwrite
    work_queue: List[Tuple[str, dict, Optional[Path], Path]] = []
    for task_id, task, init_frame in tasks:
        out_path = out_dir / f"{task_id}.mp4"
        if out_path.exists() and not overwrite:
            print(f"[{runner.name}] SKIP {task_id} (already exists)")
            continue
        work_queue.append((task_id, task, init_frame, out_path))

    print(f"\n[{runner.name}] {len(work_queue)} task(s) to generate → {out_dir}")
    if limiter:
        print(f"[{runner.name}] RPM limit: {rpm_limit} requests/min")

    results: Dict[str, bool] = {}

    def do_one(item: Tuple) -> Tuple[str, bool]:
        task_id, task, init_frame, out_path = item
        if limiter:
            limiter.acquire()
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
        "--pattern", default=None, metavar="GLOB",
        help="Glob pattern to filter task IDs, e.g. 'pouring_water_into_cup*'. "
             "If omitted, all tasks under tasks_root are processed.",
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
        "--n_gpu", type=int, default=None,
        help="Override the n_gpu value from the model config for all local models.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Stream script stdout/stderr live to the terminal (local models only).",
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
            cfg = models_config[name]
            if cfg.get("type") == "local":
                overrides = {}
                if args.n_gpu is not None:
                    overrides["n_gpu"] = args.n_gpu
                if args.verbose:
                    overrides["verbose"] = True
                if overrides:
                    cfg = {**cfg, **overrides}
            runners[name] = build_runner(name, cfg)
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

    if args.pattern:
        tasks = [(tid, t, f) for tid, t, f in tasks if fnmatch.fnmatch(tid, args.pattern)]
        if not tasks:
            print(f"[ERROR] No tasks match pattern '{args.pattern}'", file=sys.stderr)
            return 1

    print(f"Discovered {len(tasks)} task(s) under {tasks_root}")

    output_root = Path(args.output_root).expanduser().resolve()

    # ---- Run each model in sequence (tasks within a model may be parallel) ----
    overall_failed: Dict[str, List[str]] = {}

    for model_name, runner in runners.items():
        model_tasks = tasks
        if runner.camera_control_field:
            model_tasks = _dedup_for_camera_model(tasks)
            print(
                f"[{model_name}] Camera-controlled: deduplicated"
                f" {len(tasks)} → {len(model_tasks)} tasks (one per occlusion group)"
            )

        out_dir = output_root / f"{model_name}_{args.run_name}"
        results = run_model(
            runner=runner,
            model_type=models_config[model_name].get("type", "local"),
            tasks=model_tasks,
            out_dir=out_dir,
            run_name=args.run_name,
            workers=args.workers,
            overwrite=args.overwrite,
            rpm_limit=models_config[model_name].get("rpm_limit") or None,
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
