#!/usr/bin/env python3
"""
Reads a task YAML, loads:
  - input.initial_frame (image path)
  - prompts.video_gen (inline text or a .txt path)
Calls Google Gemini to generate:
  (1) init2final_edit_prompt (image-edit prompt to reach the expected final frame)
  (2) binary evaluation questions for a judge VLM

Then WRITES the result back into the SAME task YAML under:

evaluation:
  init2final_edit_prompt: ...
  rationale: ...
  questions:
    - id: q01
      question: ...
      answer_type: yes_no
      notes_for_judge: ...
  generation:
    llm: ...
    temperature: ...
    model_provider: google_gemini
    created_utc: ...
  inputs_digest:
    initial_frame_sha256: ...
    video_prompt_sha256: ...
    task_yaml_sha256_before: ...

Usage:
  python scripts/augment_task_yaml_with_eval.py \
    --task_yaml benchmark/tasks/.../task_0001.yaml \
    --model gemini-2.5-flash \
    --temperature 0.2

Env:
  export GEMINI_API_KEY="..."
"""

import argparse
import base64
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image  
from google import genai

from utils import resolve_task_paths
from utils import load_yaml, dump_yaml
from utils import sha256_bytes, sha256_file, sha256_text

# -----------------------------
# Task parsing
# -----------------------------

def read_text_maybe_path(value: str, base_dir: Path) -> str:
    candidate = (base_dir / value).resolve()
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return value

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / ".git").exists() or (cur / "README.md").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError("Could not locate repo root (no .git or README.md found).")

def get_task_fields(task: Dict[str, Any], task_path: Path) -> Tuple[str, str]:
    task_id = str(task.get("id") or task_path.stem)

    # input_block = task.get("input", {}) or {}
    # initial_frame_rel = input_block.get("initial_frame")
    # if not initial_frame_rel:
    #     raise ValueError("Missing required field: input.initial_frame")

    # # initial_frame_path = (task_path.parent / str(initial_frame_rel)).resolve()
    # # initial_frame_path = Path(initial_frame_rel)
    # repo_root = find_repo_root(task_path)
    # initial_frame_path = (repo_root / initial_frame_rel).resolve()
    # if not initial_frame_path.exists():
    #     alt = Path(str(initial_frame_rel)).expanduser().resolve()
    #     if alt.exists():
    #         initial_frame_path = alt
    #     else:
    #         raise FileNotFoundError(f"Initial frame not found: {initial_frame_path}")

    prompts_block = task.get("prompts", {}) or {}
    video_gen = prompts_block.get("video_WM")
    if not video_gen:
        raise ValueError("Missing required field: prompts.video_WM")

    video_prompt = read_text_maybe_path(str(video_gen), task_path.parent).strip()
    if not video_prompt:
        raise ValueError("Video prompt resolved to empty text.")

    # return task_id, initial_frame_path, video_prompt
    return task_id, video_prompt


# -----------------------------
# Gemini prompting
# -----------------------------

def image_to_inline_data(image_path: Path, max_side: int = 1024) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"inline_data": {"mime_type": "image/png", "data": data}}

JSON_SCHEMA_HINT = {
  "task_id": "string",
  "ground_truth_edit_prompt": "string",
  "rationale": "string",
  "binary_questions": [
    {
      "question_id": "string",
      "question": "string",
      "answer_type": "yes_no",
      "notes_for_judge": "string (optional)"
    }
  ]
}

def build_gemini_request(task_id: str, video_prompt: str) -> str:
    return f"""
You are helping build a stateful world-model benchmark.

You are given:
- an initial frame image (the starting visual state)
- a video-generation prompt describing camera motion, occlusion, and physical actions

Your job is to produce TWO artifacts with a STRICT separation of roles:

==============================
(1) init2final_edit_prompt
==============================

This must be a SINGLE prompt for an IMAGE EDITING model that edits the INITIAL FRAME
into the EXPECTED FINAL FRAME after the rollout completes.

CRITICAL RULES:
- This field must contain ONLY concrete image-editing instructions.
- Do NOT include reasoning, explanation, justification, or references to time passing.
- Do NOT mention camera motion, occlusion, or video dynamics.
- Do NOT mention timestamps, clocks, watermarks, UI overlays, subtitles, or text in the image.
- Ignore any timestamps or watermarks that may appear in example outputs; they must NOT be edited or referenced.
- Describe ONLY what the final image should look like (objects, attributes, relative configuration).
- Keep camera viewpoint and framing identical to the initial frame unless the prompt explicitly returns to a different view.

Think of this as instructions you would give to Photoshop Generative Fill.

==============================
(2) Binary evaluation questions
==============================

Produce 6-12 YES/NO questions that a judge VLM can answer by comparing:
- the initial frame
- a candidate final frame generated by a world model

Rules for questions:
- Answerable from images alone.
- Test correctness of world state implied by the rollout.
- Include at least 2 negative-control questions (things that should remain unchanged).
- Do NOT ask about timestamps, clocks, watermarks, text overlays, or UI elements.
- Do NOT ask about invisible variables or inferred time.
- Phrase questions so a correct final frame yields an unambiguous Yes or No.

==============================
(3) Rationale
==============================

Provide a short rationale (3-8 sentences) explaining:
- how the world state evolves during the rollout
- what assumptions you made about ongoing actions during occlusion

ALL reasoning, time references, and explanations MUST go here.
Do NOT repeat reasoning in the edit prompt or questions.

==============================

Return ONLY valid JSON matching this schema (no markdown, no commentary):

{JSON_SCHEMA_HINT}

Task ID: {task_id}

Video prompt:
\"\"\"{video_prompt}\"\"\"
""".strip()

def extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Could not find JSON object in Gemini response.")
    return json.loads(m.group(0))

def call_gemini(
    api_key: str,
    model: str,
    initial_frame_path: Path,
    task_id: str,
    video_prompt: str,
    temperature: float,
) -> Dict[str, Any]:
    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model=model,
        contents=[
            image_to_inline_data(initial_frame_path),
            {"text": build_gemini_request(task_id, video_prompt)},
        ],
        config={"temperature": temperature},
    )

    raw = getattr(resp, "text", None) or str(resp)
    data = extract_json(raw)

    # Normalize
    data["task_id"] = task_id
    if "rationale" not in data or not isinstance(data["rationale"], str):
        data["rationale"] = ""

    if "binary_questions" not in data or not isinstance(data["binary_questions"], list):
        raise ValueError("Gemini output missing binary_questions list.")
    for i, q in enumerate(data["binary_questions"], start=1):
        q.setdefault("answer_type", "yes_no")
        q.setdefault("question_id", f"q{i:02d}")

    return data

def upsert_evaluation_block(
    task: Dict[str, Any],
    gen: Dict[str, Any],
    *,
    model: str,
    temperature: float,
    initial_frame_sha: str,
    video_prompt_sha: str,
    task_yaml_sha_before: str,
) -> None:
    evaluation = task.get("evaluation", {}) or {}

    # Main artifacts
    evaluation["init2final_edit_prompt"] = gen["ground_truth_edit_prompt"].strip()
    if gen.get("rationale"):
        evaluation["rationale"] = gen["rationale"].strip()

    # Questions (structured)
    questions_out: List[Dict[str, Any]] = []
    for q in gen["binary_questions"]:
        questions_out.append(
            {
                "id": str(q.get("question_id", "")).strip() or "q??",
                "question": str(q.get("question", "")).strip(),
                "answer_type": "yes_no",
                "notes_for_judge": str(q.get("notes_for_judge", "")).strip(),
            }
        )
    evaluation["questions"] = questions_out

    # Provenance (helpful for auditing)
    evaluation["generation"] = {
        "model_provider": "google_gemini",
        "llm": model,
        "temperature": float(temperature),
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    evaluation["inputs_digest"] = {
        "initial_frame_sha256": initial_frame_sha,
        "video_prompt_sha256": video_prompt_sha,
        "task_yaml_sha256_before": task_yaml_sha_before,
    }

    task["evaluation"] = evaluation


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_path",
        required=True,
        type=str,
        help=(
            "Path to the task folder OR to the task YAML. "
            "Folder convention: <task_dir>/<task_dir_name>.yaml and "
            "<task_dir>/<task_dir_name>_init_frame.png"
        ),
    )
    parser.add_argument("--model", default="gemini-2.5-pro", type=str)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite evaluation fields if already present. (Default: always overwrite these keys)",
    )
    args = parser.parse_args()

    task_path = Path(args.task_path).resolve()
    _, task_yaml_path, _, initial_frame_path = resolve_task_paths(task_path)

    # Hash YAML before modification
    task_yaml_before_bytes = task_yaml_path.read_bytes()
    task_yaml_sha_before = sha256_bytes(task_yaml_before_bytes)

    task = load_yaml(task_yaml_path)
    task_id, video_prompt = get_task_fields(task, task_yaml_path)

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

    gen = call_gemini(
        api_key=api_key,
        model=args.model,
        initial_frame_path=initial_frame_path,
        task_id=task_id,
        video_prompt=video_prompt,
        temperature=args.temperature,
    )

    upsert_evaluation_block(
        task,
        gen,
        model=args.model,
        temperature=args.temperature,
        initial_frame_sha=sha256_file(initial_frame_path),
        video_prompt_sha=sha256_text(video_prompt),
        task_yaml_sha_before=task_yaml_sha_before,
    )

    # Write back into the YAML file
    dump_yaml(task_yaml_path, task)
    print(f"Updated task YAML with evaluation block: {task_yaml_path}")


if __name__ == "__main__":
    main()
