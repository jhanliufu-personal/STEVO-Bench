# Gemini-based State Evolution Evaluation

This directory contains scripts for evaluating whether expected state evolution happened in generated videos using Google's Gemini API.

## Overview

The evaluation uses a two-step Gemini-based approach:

1. **Step 1: Process Prediction**
   - Input: Initial image + optional action prompt
   - Output: Predicted expected physical process/state evolution

2. **Step 2: Process Verification**
   - Input: Generated video + expected process from Step 1
   - Output: Verdict (Yes/No) + detailed explanation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Gemini API key:
```bash
export GOOGLE_API_KEY='your-api-key-here'
```

Get your API key from: https://makersuite.google.com/app/apikey

## Usage

### Run full evaluation

Evaluate all level 4 tasks (non-_00 variants) across all 4 runs:

```bash
python run_evaluation.py
```

### Test on limited tasks

For testing, evaluate only the first N tasks:

```bash
python run_evaluation.py --limit 2
```

### Custom output filename

```bash
python run_evaluation.py --output my_results.json
```

### Custom project root

```bash
python run_evaluation.py --project-root /path/to/STEVO-Bench
```

### Parallel execution

Evaluate tasks in parallel using multiple workers (faster):

```bash
python run_evaluation.py --workers 4
```

### Evidence-based mode (recommended)

Use the evidence-based prompt (Judge 2 from ensemble testing - best single judge):

```bash
python run_evaluation.py --evidence
```

This mode achieved **81.8% accuracy** in testing, outperforming both the standard evaluator and the ensemble.

### Ensemble mode

Use ensemble evaluation with 3 different prompts and majority voting (slower but more accurate, reduces hallucinations):

```bash
python run_evaluation.py --ensemble
```

You can combine ensemble mode with parallel execution:

```bash
python run_evaluation.py --ensemble --workers 2
```

**Note:** Ensemble mode runs 3 Gemini API calls per task instead of 1, so it will be ~3x slower and use ~3x more API credits.

**Recommendation:** Use `--evidence` for best performance with standard API cost, or `--ensemble` if you need maximum robustness.

## Output

Results are saved to `eval_se/results/evaluation_results_<timestamp>.json` with the following structure:

```json
{
  "timestamp": "2024-02-26T10:30:00",
  "num_tasks_evaluated": 11,
  "summary": {
    "total": 11,
    "passed": 8,
    "failed": 3,
    "pass_rate": 0.727,
    "by_run": {
      "sora_pro_sora_pro_subset_run_3": {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "pass_rate": 0.5
      },
      ...
    }
  },
  "results": [
    {
      "task_id": "paper_burning",
      "run_name": "sora_pro_sora_pro_subset_run_3",
      "task_level": 4,
      "action_prompt": null,
      "expected_process": "...",
      "verification_response": "...",
      "verdict": true,
      "pass": true
    },
    ...
  ]
}
```

## Scripts

- `data_collector.py` - Collects task data from summary.json files and organizes resources
- `gemini_evaluator.py` - Implements two-step Gemini evaluation (standard mode)
- `gemini_evaluator_evidence.py` - Implements evidence-based evaluation (Judge 2 - best performer)
- `gemini_evaluator_ensemble.py` - Implements ensemble evaluation with 3 prompts and majority voting
- `run_evaluation.py` - Main script that orchestrates evaluation and saves results
- `analyze_results.py` - Creates confusion matrix comparing Gemini verdicts vs ground truth
- `analyze_ensemble_judges.py` - Analyzes individual judge performance in ensemble evaluation
- `pairwise_agreement.py` - Analyzes pairwise model ranking agreement with weighted scoring

## Evaluated Runs

The evaluation covers level 4 tasks (excluding _00 variants) from:
1. `sora_pro_sora_pro_subset_run_3`
2. `veo_veo_run_6`
3. `LingBot_lingbot_run_1`
4. `HY-WorldPlay_hunyuan_run_2`

## Task Filtering

Tasks are included in evaluation if:
- Task level is 4
- Task ID does NOT end with "_00"
- All required resources exist (initial image, video, YAML file)

## Notes

- Some tasks (e.g., `document_shredding`, `hydraulic_press_can_crush`) may be skipped if YAML files are missing
- Video processing may take a few seconds per task
- API rate limits may apply depending on your Gemini API tier
