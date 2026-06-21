# Runs Gemini ensemble_size=7 ablation for all five run subsets and all three judge flags.
# Existing 3-response reports are reused; only 4 new responses are generated per task.

conda activate bgm
$env:GOOGLE_API_KEY = "AIzaSyDuqkleeV3xWxep0FOykFY3tUm_okbsK7Y"

$runs = @(
    # "cogvideox_cvx_run_1"
    "HY-WorldPlay_hunyuan_run_4"
    # "LingBot_lingbot_run_2",
    # "sora_pro_sora_pro_run_4",
    # "veo_veo_run_7",
    # "wan22_wan_run_2"
)

$flags = @("--control", "--physics", "--state")

$total = $runs.Count * $flags.Count
$i = 0

foreach ($run in $runs) {
    foreach ($flag in $flags) {
        $i++
        $ts = Get-Date -Format "HH:mm:ss"
        Write-Host "[$ts] ($i/$total) $flag  $run"

        $pyArgs = @(
            "-m", "eval.eval_cli",
            "--outputs", ".\outputs\$run",
            # "--task_root", ".\benchmark\tasks_subset\",
            # "--run_dir", ".\runs_subset\",
            "--task_root", ".\benchmark\tasks\",
            "--run_dir", ".\runs\",
            "--vlm_provider", "gemini",
            "--control_judge_model", "gemini-3.1-pro-preview",
            "--physics_judge_model", "gemini-3.1-pro-preview",
            "--se_judge_model", "gemini-3.1-pro-preview",
            $flag,
            "--ensemble_size", "3",
            "--workers", "3",
            "--exclude_baseline"
        )
        & python @pyArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Non-zero exit ($LASTEXITCODE) for $flag $run"
        }
    }
}

Write-Host "Done. $total jobs completed."
