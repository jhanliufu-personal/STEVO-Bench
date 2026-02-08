#!/usr/bin/env bash
set -euo pipefail

# ================================
# Usage:
#   ./batch_generate_gt_and_questions.sh <TASKS_ROOT_DIR> [PYTHON_BIN]
#
# Example:
#   ./batch_generate_gt_and_questions.sh benchmark/tasks
#   ./batch_generate_gt_and_questions.sh benchmark/tasks python3
# ================================

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <TASKS_ROOT_DIR> [PYTHON_BIN]"
  exit 1
fi

TASKS_ROOT="$(realpath "$1")"
PYTHON_BIN="${2:-python}"

SCRIPT_PATH="benchmark/runners/build_benchmark/generate_gt_and_questions.py"

if [[ ! -d "$TASKS_ROOT" ]]; then
  echo "Error: TASKS_ROOT_DIR does not exist: $TASKS_ROOT"
  exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: Python script not found: $SCRIPT_PATH"
  exit 1
fi

echo "Running world-model benchmark GT generation"
echo "Tasks root: $TASKS_ROOT"
echo "Python: $PYTHON_BIN"
echo "--------------------------------------------"

FOUND_ANY=false

# Find all directories (depth >= 1)
while IFS= read -r -d '' TASK_DIR; do
  BASENAME="$(basename "$TASK_DIR")"
  YAML_PATH="$TASK_DIR/$BASENAME.yaml"
  INIT_FRAME="$TASK_DIR/${BASENAME}_init_frame.png"

  # Check naming convention
  if [[ ! -f "$YAML_PATH" ]]; then
    echo "[SKIP] $TASK_DIR (missing $BASENAME.yaml)"
    continue
  fi

  if [[ ! -f "$INIT_FRAME" ]]; then
    echo "[SKIP] $TASK_DIR (missing ${BASENAME}_init_frame.png)"
    continue
  fi

  FOUND_ANY=true
  echo
  echo "[RUN ] Task: $TASK_DIR"

  set +e
  $PYTHON_BIN "$SCRIPT_PATH" \
    --task_path "$TASK_DIR"
  STATUS=$?
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "[FAIL] Task failed: $TASK_DIR (exit code $STATUS)"
  else
    echo "[DONE] Task completed: $TASK_DIR"
  fi

done < <(find "$TASKS_ROOT" -type d -print0)

if [[ "$FOUND_ANY" = false ]]; then
  echo "No valid task directories found under: $TASKS_ROOT"
  exit 1
fi

echo
echo "Generated GT final frame and questions for all tasks"
