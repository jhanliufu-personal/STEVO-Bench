#!/usr/bin/env bash
# run_eval.sh
#
# Run all evaluation criteria for a single model output folder.
# Each criterion uses the ensemble configuration from the paper.
#
# Usage:
#   bash run_eval.sh <outputs_folder> [run_dir]
#
# Arguments:
#   outputs_folder   Path to the model output folder (must contain exactly one
#                    .json output map and the corresponding video files).
#   run_dir          Directory to write evaluation results (default: runs/).
#
# Example:
#   bash run_eval.sh outputs/veo_veo_run_1
#   bash run_eval.sh outputs/veo_veo_run_1 my_runs/

set -euo pipefail

OUTPUTS="${1:?Error: outputs_folder is required. Usage: bash run_eval.sh <outputs_folder> [run_dir]}"
RUN_DIR="${2:-runs/}"

TASK_ROOT="benchmark/tasks"
WORKERS=5

echo "========================================"
echo "Evaluating: $OUTPUTS"
echo "Results → $RUN_DIR"
echo "========================================"

# ---------------------------------------------------------------------------
# Control (occlusion_done + trigger_applied)
# Ensemble: majority
# ---------------------------------------------------------------------------
echo ""
echo "=== [1/4] Control ==="
python -m eval.eval_cli \
    --outputs "$OUTPUTS" \
    --task_root "$TASK_ROOT" \
    --run_dir "$RUN_DIR" \
    --control \
    --ensemble_size 3 \
    --ensemble_mode majority \
    --workers "$WORKERS"

# ---------------------------------------------------------------------------
# Physics / Artifact
# Ensemble: majority
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/4] Physics ==="
python -m eval.eval_cli \
    --outputs "$OUTPUTS" \
    --task_root "$TASK_ROOT" \
    --run_dir "$RUN_DIR" \
    --artifact \
    --ensemble_size 3 \
    --ensemble_mode majority \
    --workers "$WORKERS"

# ---------------------------------------------------------------------------
# Coherence
# Ensemble: unanimous_true (avoids penalizing coherent videos when one judge
# makes a false-negative error)
# ---------------------------------------------------------------------------
echo ""
echo "=== [3/4] Coherence ==="
python -m eval.eval_cli \
    --outputs "$OUTPUTS" \
    --task_root "$TASK_ROOT" \
    --run_dir "$RUN_DIR" \
    --coherence \
    --ensemble_size 3 \
    --ensemble_mode unanimous_true \
    --workers "$WORKERS"

# ---------------------------------------------------------------------------
# State Evolution
# Ensemble: majority
# ---------------------------------------------------------------------------
echo ""
echo "=== [4/4] State Progress ==="
python -m eval.eval_cli \
    --outputs "$OUTPUTS" \
    --task_root "$TASK_ROOT" \
    --run_dir "$RUN_DIR" \
    --state \
    --ensemble_size 3 \
    --ensemble_mode majority \
    --workers "$WORKERS"

echo ""
echo "========================================"
echo "Done. Results written to: $RUN_DIR"
echo "========================================"
