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
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from eval.judge_client import JudgeClient, make_judge_client
from eval.task_resolver import ResolvedTask


# -----------------------------
# Data structures
# -----------------------------

AnswerValue = Literal["yes", "no", "unknown"]


@dataclass
class JudgeAnswer:
    id: str
    answer: AnswerValue
    confidence: Optional[float] = None
    notes: str = ""


@dataclass
class JudgeResult:
    task_id: str
    provider: str
    model: str

    init_frame: str
    final_frame: str

    raw_text: str
    answers: List[JudgeAnswer]

    report_path: str


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
    lines.append("Return your answers in this JSON format ONLY (no extra text):")
    lines.append('{"answers":[{"id":"q01","answer":"yes|no|unknown","confidence":0.0-1.0,"notes":"optional"}]}')
    lines.append("")
    lines.append("Questions:")

    for q in task.questions:
        lines.append(f"- {q.id}: {q.question}")

    return "\n".join(lines).strip()


# -----------------------------
# Parsing
# -----------------------------

def _normalize_answer(s: str) -> AnswerValue:
    s = s.strip().lower()
    if s in ("yes", "y", "true"):
        return "yes"
    if s in ("no", "n", "false"):
        return "no"
    return "unknown"


# def parse_judge_output(raw_text: str, expected_ids: List[str]) -> List[JudgeAnswer]:
#     """
#     Parses judge output. Primary path: JSON.
#     Fallback path: regex line matching like 'q01: yes'.

#     Keeps it intentionally simple.
#     """
#     raw = (raw_text or "").strip()
#     answers: Dict[str, JudgeAnswer] = {}

#     # Try JSON first
#     try:
#         data = json.loads(raw)
#         items = data.get("answers", [])
#         if isinstance(items, list):
#             for it in items:
#                 qid = str(it.get("id", "")).strip()
#                 if not qid:
#                     continue
#                 ans = _normalize_answer(str(it.get("answer", "")))
#                 conf = it.get("confidence", None)
#                 try:
#                     conf_f = float(conf) if conf is not None else None
#                 except Exception:
#                     conf_f = None
#                 notes = str(it.get("notes", "")).strip()
#                 answers[qid] = JudgeAnswer(id=qid, answer=ans, confidence=conf_f, notes=notes)
#     except Exception:
#         pass

#     # Fallback: parse lines like "q01: yes"
#     if not answers:
#         for line in raw.splitlines():
#             m = re.match(r"^\s*([A-Za-z0-9_\-]+)\s*[:=]\s*(yes|no|unknown)\b", line.strip(), re.IGNORECASE)
#             if not m:
#                 continue
#             qid = m.group(1).strip()
#             ans = _normalize_answer(m.group(2))
#             answers[qid] = JudgeAnswer(id=qid, answer=ans)

#     # Ensure all expected are present (fill unknown)
#     out: List[JudgeAnswer] = []
#     for qid in expected_ids:
#         out.append(answers.get(qid, JudgeAnswer(id=qid, answer="unknown")))

#     return out

def _strip_code_fences_if_entire_block(s: str) -> str:
    """
    If the *entire* string is a fenced block, unwrap it.
    Otherwise leave it (we'll extract JSON later).
    """
    s = (s or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s

def _extract_json_candidate(s: str) -> Optional[str]:
    """
    Extract JSON even if the model prints reasoning before/after.
    Priority:
      1) JSON inside a ```json ... ``` (or ``` ... ```) fenced block
      2) First {...} object anywhere in the text
    """
    s = (s or "").strip()

    # 1) Prefer a fenced block if present (may not be the entire string)
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        # inner might still contain extra text; try to pull first object
        obj = re.search(r"\{.*\}", inner, flags=re.DOTALL)
        return (obj.group(0).strip() if obj else inner)

    # 2) Otherwise, pull first JSON object from anywhere
    obj = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if obj:
        return obj.group(0).strip()

    return None

def parse_judge_output(raw_text: str, expected_ids: List[str]) -> List[JudgeAnswer]:
    raw = (raw_text or "").strip()

    answers: Dict[str, JudgeAnswer] = {}

    # Try extracting JSON from anywhere (handles reasoning before fenced JSON)
    candidate = _extract_json_candidate(raw)

    if candidate:
        # If candidate is itself a fenced-only string, unwrap (rare)
        candidate = _strip_code_fences_if_entire_block(candidate)

        try:
            data = json.loads(candidate)
        except Exception:
            data = None

        if isinstance(data, dict):
            items = data.get("answers", [])
            if isinstance(items, list):
                for it in items:
                    qid = str(it.get("id", "")).strip()
                    if not qid:
                        continue
                    ans = _normalize_answer(str(it.get("answer", "")))
                    conf = it.get("confidence", None)
                    try:
                        conf_f = float(conf) if conf is not None else None
                    except Exception:
                        conf_f = None
                    notes = str(it.get("notes", "")).strip()
                    answers[qid] = JudgeAnswer(id=qid, answer=ans, confidence=conf_f, notes=notes)

    # Fallback: parse lines like "q01: yes"
    if not answers:
        for line in raw.splitlines():
            m = re.match(
                r"^\s*([A-Za-z0-9_\-]+)\s*[:=]\s*(yes|no|unknown)\b",
                line.strip(),
                re.IGNORECASE,
            )
            if not m:
                continue
            qid = m.group(1).strip()
            ans = _normalize_answer(m.group(2))
            answers[qid] = JudgeAnswer(id=qid, answer=ans)

    # Fill missing with unknown (preserve expected order)
    out: List[JudgeAnswer] = []
    for qid in expected_ids:
        out.append(answers.get(qid, JudgeAnswer(id=qid, answer="unknown")))

    return out


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

    raw = client.judge(prompt, init_image=task.init_frame, final_image=task.final_frame)

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
        "prompt": prompt,
        "raw_text": raw,
        "answers": [asdict(a) for a in answers],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return JudgeResult(
        task_id=task.task_id,
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
