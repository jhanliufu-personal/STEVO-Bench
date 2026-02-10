import yaml  
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple

# -----------------------------
# YAML IO (preserve readability)
# -----------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    # Keep output pretty and stable for diffs
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            width=100,
            default_flow_style=False,
        )

# -----------------------------
# Hashing / digests
# -----------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))

def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

# -----------------------------
# Path resolution
# -----------------------------

def _find_task_yaml_in_dir(task_dir: Path) -> Path:
    """By convention <task_dir>/<task_dir_name>.yaml"""
    folder = task_dir.name
    y = task_dir / f"{folder}.yaml"
    if not y.exists():
        raise FileNotFoundError(f"Missing task YAML: {y}")
    return y

def _find_init_frame_in_dir(task_dir: Path) -> Path:
    """By convention <task_dir>/<task_dir_name>_init_frame.png"""
    folder = task_dir.name
    img = task_dir / f"{folder}_init_frame.png"
    if not img.exists():
        raise FileNotFoundError(f"Missing init frame: {img}")
    return img

def resolve_task_paths(task_dir: Path) -> Tuple[Path, Path, str, Path]:
    """
    Expects task_dir to be a directory.

    Convention:
      <task_dir>/<task_dir_name>.yaml
      <task_dir>/<task_dir_name>_init_frame.png

    Returns:
      task_dir, task_yaml_path, folder_name, init_frame_path
    """
    task_dir = Path(task_dir).expanduser().resolve()
    if not task_dir.exists():
        raise FileNotFoundError(task_dir)
    if not task_dir.is_dir():
        raise ValueError(f"Expected a task directory, got file: {task_dir}")

    folder_name = task_dir.name
    task_yaml_path = task_dir / f"{folder_name}.yaml"
    init_frame_path = task_dir / f"{folder_name}_init_frame.png"

    if not task_yaml_path.exists():
        raise FileNotFoundError(f"Task YAML not found: {task_yaml_path}")
    if not init_frame_path.exists():
        raise FileNotFoundError(f"Init frame not found: {init_frame_path}")

    return task_dir, task_yaml_path, folder_name, init_frame_path