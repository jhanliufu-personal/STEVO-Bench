# Benchmark Runners

Scripts for generating the StateWMBench benchmark. All scripts require `GOOGLE_API_KEY` in the environment.

## Generation Pipeline

Tasks are generated in this order:

```
generate_task_variants.py   →   generate_init_frame.py   →   generate_gt_and_questions.py   →   generate_gt_final_frame.py
```

Use `batch_generate.py` to run any of the per-task scripts across an entire directory tree in parallel.

---

## Scripts

### `generate_task_variants.py`
Given a base task YAML and its initial frame, calls Gemini to produce a family of variants along two axes:
1. **Occlusion method** – replaces camera rotation with alternatives (object blocking, zoom-out, lighting change, etc.)
2. **Dynamic conditions** – changes physical parameters (temperature, friction, mass, etc.) to alter the expected outcome

Always creates a fully-observable baseline variant (`_00`) first (no occlusion, stationary camera). Saves each variant as a new task directory with its own YAML. Accepts `--task_path` (single task) or `--level_dir` (all tasks in a level).

```bash
python generate_task_variants.py --task_path benchmark/tasks/level1_scalar_state/ice_on_burner_01 --num_variants 10
```

---

### `generate_init_frame.py`
Generates or edits the initial frame image for a task using the Gemini image model (Nano Banana API):
- **No frame exists** → generates from `prompts.image_gen_prompt`
- **Frame belongs to a different task** (variant) → edits it using `prompts.init_frame_edit_prompt`; if the prompt is `[NO CHANGE]`, copies and renames the original
- **Frame already exists** → skips unless `--overwrite`

```bash
python generate_init_frame.py --task_path benchmark/tasks/level1_scalar_state/ice_on_burner_01_02
```

---

### `generate_gt_and_questions.py`
Calls Gemini with the initial frame and `video_WM` prompt to produce the evaluation block for a task YAML:
- `evaluation.init2final_edit_prompt` – image-edit prompt to turn the initial frame into the expected final state
- `evaluation.rationale` – explanation of the expected state evolution
- `evaluation.questions` – 6–12 binary yes/no questions for a judge VLM, with level-specific guidance (Levels 0–5)
- `evaluation.generation` / `evaluation.inputs_digest` – provenance metadata

Writes the result back into the same YAML file.

```bash
python generate_gt_and_questions.py --task_path benchmark/tasks/level1_scalar_state/ice_on_burner_01
```

---

### `generate_gt_final_frame.py`
Takes the `evaluation.init2final_edit_prompt` from a task YAML and calls the Gemini image-editing model to produce `<task_id>_gt_final_frame.png` in the task directory. Skips if the file already exists unless `--overwrite`.

```bash
python generate_gt_final_frame.py --task_path benchmark/tasks/level1_scalar_state/ice_on_burner_01
```

---

### `batch_generate.py`
Runs any of the per-task scripts above over a directory tree, skipping directories without a matching YAML. Supports parallel execution via `--workers`.

Change the `script_name` variable at the top of the file to select which script to batch-run.

```bash
python batch_generate.py --tasks_root_dir benchmark/tasks/level1_scalar_state --workers 4
```

---

### `cleanup_tasks.py`
Resets task directories for fresh regeneration by removing PNG files and/or `evaluation` blocks from YAMLs. Use `--dry-run` to preview, `--confirm` to skip the interactive prompt.

```bash
python cleanup_tasks.py --tasks_root benchmark/tasks/level1_scalar_state --dry-run
python cleanup_tasks.py --tasks_root benchmark/tasks/level1_scalar_state --confirm
```

---

### `utils.py`
Shared helpers used by all scripts above:
- YAML load/dump (human-readable, diff-stable)
- SHA-256 hashing for provenance digests
- `resolve_task_paths` – resolves a task directory to its YAML and init frame by naming convention
- `image_to_inline_data` – encodes images for the Gemini API
- `extract_json` – parses JSON from Gemini responses (handles markdown code fences)
- `call_nanobanana` – unified wrapper for Gemini image generation and editing
