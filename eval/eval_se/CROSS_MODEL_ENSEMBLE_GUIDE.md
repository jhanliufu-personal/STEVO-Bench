# Cross-Model Ensemble Evaluator Guide

## Overview

The cross-model ensemble evaluator (`gemini_gpt_evaluator.py`) combines **Gemini** and **GPT** (OpenAI) to evaluate state evolution in videos. It uses **unanimous voting** - both models must agree for a task to PASS.

## Key Features

1. **Two-model consensus**: Uses both Gemini-3-pro-preview and GPT-5.2
2. **Parallel execution**: Sends requests to both APIs simultaneously for speed
3. **Same prompts**: Uses identical prompts from `gemini_evaluator.py`
4. **Unanimous voting**: Only marks PASS if BOTH models agree
5. **More conservative**: Should have higher precision, lower false positives
6. **Full transparency**: Saves both models' responses and verdicts
7. **Hybrid approach**: Gemini analyzes full video, GPT analyzes extracted frames (every 5th)

## Setup

### 1. Install dependencies

```bash
pip install -r eval_se/requirements.txt
```

This installs:
- `openai>=1.0.0` - OpenAI API client for GPT-5.2
- `opencv-python>=4.8.0` - For video frame extraction (GPT side)

### 2. Set API keys

You need BOTH API keys:

```bash
export GOOGLE_API_KEY='your-gemini-key'
export OPENAI_API_KEY='your-openai-key'
```

## Usage

### Run evaluation with cross-model ensemble

```bash
# Test on 5 tasks
python eval_se/run_evaluation.py --cross-model --limit 5

# Run on all tasks
python eval_se/run_evaluation.py --cross-model

# With custom output filename
python eval_se/run_evaluation.py --cross-model --output my_results.json
```

### Flags

- `--cross-model`: Enable cross-model ensemble (Gemini + GPT)
- `--limit N`: Limit to N tasks (for testing)
- `--output FILE`: Custom output filename
- `--workers N`: Number of parallel workers (default: 1)

**Note**: Cannot use `--cross-model` with `--ensemble` or `--evidence` flags.

## How It Works

### Step 1: Process Prediction (Gemini only)
Gemini predicts the expected process from the initial image:
- Uses same prompt as `gemini_evaluator.py`
- **Only Gemini** generates the prediction
- This prediction is used for verification in Step 2

### Step 2: Video Verification (Parallel - Gemini + GPT)
Both models verify if the process happened in the video:
- **Separate prompts**: Gemini and GPT use separate verification prompts (customizable)
- **Gemini**: Receives full video via API upload
- **GPT**: Receives every 5th frame as concatenated images
- **Both API calls made in parallel** for speed
- Both models analyze the video (full vs frames) simultaneously
- Each model gives its own verdict (Yes/No)
- **Final verdict = PASS only if BOTH say Yes**
- **Rate limiting**: Max 20 concurrent Gemini calls, max 3 concurrent GPT calls
- **Retry logic**: GPT retries up to 3 times on 400 errors with 1s delay

## Output Format

Results include both models' verification responses:

```json
{
  "task_id": "bowling_inertia",
  "run_name": "sora_pro_sora_pro_subset_run_3",
  "prediction_gemini": "Gemini's full prediction...",
  "expected_process_concise": "The bowling ball will...",
  "verification_results": {
    "gemini_response": "Gemini's verification response...",
    "gemini_verdict": true,
    "gemini_time": 15.2,
    "gpt_response": "GPT's verification response...",
    "gpt_verdict": false,
    "gpt_time": 3.1,
    "consensus_verdict": false
  },
  "verdict": false,
  "pass": false
}
```

## Expected Results

### Compared to single-model evaluators:

- **Higher Precision**: Fewer false positives (both models must hallucinate)
- **Lower Recall**: More false negatives (stricter requirements)
- **Better Correlation**: Should align better with human judgments
- **Fewer Passes**: More conservative than majority voting

### Comparison with other evaluators:

| Evaluator | Voting Method | Expected Precision | Expected Recall |
|-----------|--------------|-------------------|-----------------|
| Standard (Gemini only) | Single model | Medium | Medium |
| Evidence (Judge 2) | Single model | High | Medium |
| Ensemble (3 Gemini prompts) | Unanimous (all 3) | Very High | Low |
| **Cross-Model (Gemini+GPT)** | **Unanimous (both)** | **Very High** | **Medium-Low** |

## Analysis

After running evaluation, analyze with standard tools:

```bash
# Confusion matrix
python eval_se/analyze_results.py evaluation_results_TIMESTAMP.json

# Model win rates
python eval_se/model_win_rates.py evaluation_results_TIMESTAMP.json

# Pairwise agreement
python eval_se/pairwise_agreement.py evaluation_results_TIMESTAMP.json
```

## Cost Considerations

**Cross-model ensemble is more expensive** because it calls TWO APIs for verification:

- Gemini API calls: 2 per task (prediction + verification)
- OpenAI API calls: 1 per task (verification only)
- **Total**: 3 API calls per task

For 119 tasks:
- ~240 Gemini API calls (prediction + verification)
- ~120 OpenAI API calls (GPT-5.2 with frame extraction, verification only)

**Note**: Parallel execution reduces time but NOT cost - you still make the same number of API calls.
**Retry logic**: GPT may make additional API calls if retrying on 400 errors (up to 3 attempts per call).

## Troubleshooting

### Error: OPENAI_API_KEY not set
```bash
export OPENAI_API_KEY='your-key-here'
```

### Error: GOOGLE_API_KEY not set
```bash
export GOOGLE_API_KEY='your-key-here'
```

### Video too large
If videos are very large, they may exceed API limits:
- **Gemini**: Handles large video files directly (check API limits)
- **GPT**: Uses frame extraction (every 5th frame), so video size less critical
- Consider compressing videos if Gemini upload times are slow
- More frames = higher API cost for GPT (but better temporal resolution)

## Future Improvements

1. **Additional models**: Could add Claude, Llama Vision, etc.
2. **Weighted voting**: Give different weights to different models based on performance
3. **Confidence scores**: Use model confidence levels in voting
4. **Multi-judge per model**: Combine with 3-prompt ensemble for each model
