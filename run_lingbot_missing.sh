#!/usr/bin/env bash
# Generate missing LingBot videos for 5 tasks.
# Run from the repo root: bash run_lingbot_missing.sh
# Assumes generation/configs/models.yaml has repo_path filled in.

set -euo pipefail

RUN_NAME="lingbot_run_2"
TASKS_ROOT="benchmark/tasks/"
OUTPUT_ROOT="outputs/"
ACTIONS_DIR="generation/world_models/lingbot_actions"
# Path to the HY-WorldPlay repo's hyvideo package (needed for pose precomputation).
HYWORLD_REPO="<path_to_your_HY-WorldPlay_repo>/hyvideo"

# --- Step 1: Precompute LingBot action folders ---
echo "=== Precomputing LingBot actions ==="
python -m generation.world_models.precompute_lingbot_actions \
    --tasks_root "$TASKS_ROOT" \
    --output_dir "$ACTIONS_DIR" \
    --hyworld_repo "$HYWORLD_REPO"

# --- Step 2: Generate each missing task ---
# inflating_balloon_09 has two occlusion variants (_cardboard, _curtain);
# the glob ensures both are discovered and deduplicated to one LingBot video.
TASKS=(
    "drinking_fountain_button"
    "inflating_balloon_09*"
    "lightbulb_and_switch"
    "lightbulb_and_switch_opposite"
    "traffic_light_green"
)

echo ""
echo "=== Generating videos (run: $RUN_NAME) ==="
for pattern in "${TASKS[@]}"; do
    echo "--- pattern: $pattern ---"
    python -m generation.run_world_models \
        --models LingBot \
        --tasks_root "$TASKS_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --run_name "$RUN_NAME" \
        --pattern "$pattern"
done

echo ""
echo "Done. Outputs written to ${OUTPUT_ROOT}LingBot_${RUN_NAME}/"
