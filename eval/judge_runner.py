# eval/judge_runner.py
"""
judge_runner.py

Runs VLM judging for output-first evaluation.

Assumptions:
- task_resolver has already populated a run directory and returned ResolvedTask objects.
- Each ResolvedTask points to:
    - init_frame.png
    - final_frame.png
    - evaluation questions (yes/no)
- This module:
    1) builds a single prompt for the judge VLM
    2) calls a JudgeClient (OpenAI/Anthropic/Gemini) with init+final images
    3) parses the judge response into structured answers
    4) writes a per-task judge report into the task's run folder
    5) returns JudgeResult objects for downstream scoring

Dependencies:
- eval/judge_client.py provides make_judge_client(provider, model)
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Literal

from eval.judge_client import JudgeClient, make_judge_client
from eval.task_resolver import ResolvedTask
from eval.judge_output_parser import parse_judge_output, JudgeResult

# -----------------------------
# Prompt formatting
# -----------------------------

def build_judge_prompt(task: ResolvedTask) -> str:
    """
    Builds a single text prompt instructing the VLM judge to answer the task's questions.

    The judge() call will attach:
      - Image 1: initial frame
      - Image 2: candidate final frame

    Keep this stable; changing it changes evaluation.
    """
    lines: List[str] = []
    lines.append("You are a visual judge for a world-model benchmark.")
    lines.append("")
    lines.append("You will be given TWO images in order:")
    lines.append("Image 1 = INITIAL frame (starting state).")
    lines.append("Image 2 = FINAL frame (candidate ending state).")
    lines.append("")
    lines.append("Answer the following YES/NO questions by comparing Image 2 against Image 1.")
    lines.append("If the question cannot be determined from the images, answer UNKNOWN.")
    lines.append("")
    lines.append("Important rules:")
    lines.append("- Ignore any timestamps, watermarks, subtitles, or text overlays if present.")
    lines.append("- Base answers only on visible content.")
    lines.append("")
    # lines.append("Return your answers in this JSON format ONLY (no extra text):")
    # lines.append('{"answers":[{"id":"q01","answer":"yes|no|unknown","confidence":0.0-1.0,"notes":"optional"}]}')
    lines.append("Return your answers in this JSON format ONLY (no extra text):")
    lines.append(
        '{"answers": ['
        '{"id": "<EXACT_QUESTION_ID>", '
        '"answer": "yes|no|unknown", '
        '"confidence": 0.0-1.0, '
        '"notes": "optional"}'
        ']}'
    )
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append("- You MUST copy the question ID EXACTLY as provided in the question list.")
    lines.append("- Do NOT rename, shorten, or reformat the IDs.")
    lines.append("- Every question ID must appear exactly once in the output.")
    lines.append("")
    lines.append("Questions:")

    for q in task.questions:
        lines.append(f"- {q.id}: {q.question}")

    return "\n".join(lines).strip()

# -----------------------------
# Runner
# -----------------------------

def judge_one_task(
    task: ResolvedTask,
    *,
    provider: Literal["openai", "anthropic", "gemini"],
    model: str,
    report_filename: str = "judge_report.json",
) -> JudgeResult:
    """
    Judge a single task and write report into task.run_task_dir (per_task/<task_id>/).

    Returns a JudgeResult for downstream scoring.
    """
    client: JudgeClient = make_judge_client(provider=provider, model=model)

    prompt = build_judge_prompt(task)

    # print(prompt)

    raw = client.judge(prompt, init_image=task.init_frame, final_image=task.final_frame)
    # raw = client.judge(prompt, init_image=task.init_frame, final_image=task.gt_final_frame)

    expected_ids = [q.id for q in task.questions]
    answers = parse_judge_output(raw, expected_ids)

    # Save report
    report_path = Path(task.final_frame).parent / report_filename

    payload = {
        "task_id": task.task_id,
        "judge_provider": provider,
        "judge_model": model,
        "init_frame": str(task.init_frame),
        "final_frame": str(task.final_frame),
        # "final_frame": str(task.gt_final_frame),
        "prompt": prompt,
        "raw_text": raw,
        "answers": [asdict(a) for a in answers],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return JudgeResult(
        task_id=task.task_id,
        task_level=task.task_level,
        provider=provider,
        model=model,
        init_frame=str(task.init_frame),
        final_frame=str(task.final_frame),
        raw_text=raw,
        answers=answers,
        report_path=str(report_path),
    )


def judge_all_tasks(
    tasks: List[ResolvedTask],
    *,
    provider: Literal["openai", "anthropic", "gemini"],
    model: str,
    report_filename: str = "judge_report.json",
) -> List[JudgeResult]:
    """
    Convenience wrapper for multiple tasks.
    """
    results: List[JudgeResult] = []
    for t in tasks:
        results.append(
            judge_one_task(
                t,
                provider=provider,
                model=model,
                report_filename=report_filename,
            )
        )
    return results
