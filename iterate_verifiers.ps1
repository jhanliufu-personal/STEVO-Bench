# run_eval.ps1
conda activate bgm

$metric="coherence"

# $folders = @(
#     "outputs/sora_pro_sora_pro_subset_run_3",
#     "outputs/wan22_wan_run_1",
#     "outputs/veo_veo_run_6",
#     "outputs/LingBot_lingbot_run_1",
#     "outputs/HY-WorldPlay_hunyuan_run_2",
#     "outputs/genie_genie_run_1"
# )

# foreach ($dir in $folders) {
#     Write-Host "=== $dir ==="
#     python -m eval.eval_cli `
#         --outputs $dir `
#         --task_root benchmark/tasks_v4 `
#         --run_dir runs/ `
#         --exclude_baseline `
#         --$metric `
#         --overwrite `
#         --ensemble_size 5 `
#         --ensemble_mode "unanimous_true" `
#         --workers 10 
# }

$run_dirs = @(
    "runs/sora_pro_output_map_sora_pro_subset_run_3",
    "runs/wan22_output_map_wan_run_1",
    "runs/veo_output_map_veo_run_6",
    "runs/genie_output_map_genie_run_1",
    "runs/HY-WorldPlay_output_map_hunyuan_run_2",
    "runs/LingBot_output_map_lingbot_run_1"
)

$script="analyze_llm_human_agreement"

python -m eval.$script `
    --run_dir $run_dirs `
    --metric $metric