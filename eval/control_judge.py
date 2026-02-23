# eval/control_judge.py
"""
Gemini-only controllability evaluator.

Input:
- ResolvedTask (assumes task_resolver already ran)
- Uses:
    - task.task_yaml to load prompts.video_WM
    - task.wm_video as the WM output video

Output (per task):
- writes per_task/<task_id>/control_report.json

Returns:
- ControlJudgeResult for downstream summary

Deps:
  pip install pyyaml google-genai
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from eval.control_judge_client import make_control_judge_client
from eval.task_resolver import ResolvedTask


@dataclass
class ControlJudgeResult:
    task_id: str
    provider: str
    model: str
    report_path: str

    requested_occlusion: str
    requested_trigger: str
    occlusion_done: bool
    trigger_applied: bool

    raw_text: str


def _load_video_wm_prompt(task_yaml_path: Path) -> str:
    data = yaml.safe_load(Path(task_yaml_path).read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts", {}) or {}
    s = prompts.get("video_WM", "")
    return s.strip() if isinstance(s, str) and s.strip() else "none"


def build_control_judge_prompt(video_wm_prompt: str) -> str:
    return (
        "You are evaluating controllability of a generated video.\n\n"

        "You are given:\n"
        "1) A REQUESTED video-generation prompt (what SHOULD happen)\n"
        "2) The GENERATED video (what DID happen)\n\n"

        "Your job:\n"
        "A) From the REQUESTED prompt, extract:\n"
        "   - requested_occlusion: a concise description of the intended occlusion method\n"
        "     (the technique used to hide the relevant scene area during the trigger).\n"
        "     Occlusion methods include, but are not limited to:\n"
        "       - camera motion (pan, lean, zoom out, tilt away)\n"
        "       - lights turning off (scene goes dark)\n"
        "       - an object moving in front of the relevant area\n"
        "       - the subject moving out of frame\n"
        "     Set to \"none\" if no occlusion is described.\n"
        "   - requested_trigger: a concise description of the physical action or trigger\n"
        "     that should cause the dynamic (or \"none\")\n\n"

        "B) From the GENERATED video, decide TWO INDEPENDENT things:\n"
        "   - occlusion_done: whether the requested occlusion was achieved\n"
        "     (the relevant area was hidden from view by whatever means the prompt specified)\n"
        "   - trigger_applied: whether the requested trigger/action was applied\n"
        "     (based on visible evidence OR its physical effect)\n\n"

        "CRITICAL DECOUPLING RULE (VERY IMPORTANT):\n"
        "- Occlusion compliance and trigger execution must be evaluated separately.\n"
        "- If the prompt describes the trigger as happening \"while X is out of view\"\n"
        "  or under some occlusion condition, DO NOT treat that condition as a\n"
        "  prerequisite for trigger_applied.\n"
        "- trigger_applied should be TRUE if there is clear visual evidence that:\n"
        "    (a) the action itself occurred (e.g., switch toggled, valve turned, object released), OR\n"
        "    (b) the intended physical effect occurred (e.g., light turns off, object moves, water flows),\n"
        "  even if the occlusion was not perfectly achieved.\n"
        "- occlusion_done should reflect only whether the requested occlusion method\n"
        "  successfully hid the relevant area at the appropriate moment.\n\n"

        "Evaluation guidance:\n"
        "- Use only visual evidence from the video.\n"
        "- Prefer causal evidence over strict timing or framing constraints.\n"
        "- Ignore timestamps, watermarks, subtitles, and UI overlays.\n"
        "- If the prompt does not specify a trigger, set requested_trigger to \"none\" and trigger_applied to true.\n"
        "- If the prompt does not specify an occlusion method, set requested_occlusion to \"none\" and occlusion_done to true.\n\n"

        "Return ONLY valid JSON in this exact format (no markdown, no commentary):\n"
        "{\n"
        "  \"requested_occlusion\": \"string\",\n"
        "  \"requested_trigger\": \"string\",\n"
        "  \"occlusion_done\": true/false,\n"
        "  \"trigger_applied\": true/false,\n"
        "  \"notes\": \"optional short explanation\"\n"
        "}\n\n"

        "REQUESTED VIDEO PROMPT:\n"
        f"{video_wm_prompt.strip()}\n"
    )


# ---- robust JSON parsing (handles reasoning / fenced blocks / pure JSON) ----

def _unwrap_full_fence(s: str) -> str:
    s = (s or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s

def _extract_first_object(s: str) -> Optional[str]:
    m = re.search(r"\{.*\}", s or "", flags=re.DOTALL)
    return m.group(0).strip() if m else None

def _try_load_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def parse_control_judge_output(raw_text: str) -> Dict[str, Any]:
    raw = (raw_text or "").strip()

    # 1) pure JSON or fully fenced JSON
    data = _try_load_json(_unwrap_full_fence(raw))
    if data is None:
        data = _try_load_json(raw)

    # 2) extract first {...} anywhere (handles reasoning before JSON)
    if data is None:
        obj = _extract_first_object(raw)
        if obj:
            data = _try_load_json(_unwrap_full_fence(obj))

    if data is None:
        raise ValueError(f"Could not parse control judge JSON. Raw (truncated):\n{raw[:2000]}")
    return data


def evaluate_control_one_task(
    task: ResolvedTask,
    *,
    model: str = "gemini-3-pro-preview",
    report_filename: str = "control_report.json",
) -> ControlJudgeResult:
    client = make_control_judge_client(model=model)
    # print('Control judge client made')

    video_wm_prompt = _load_video_wm_prompt(Path(task.task_yaml))
    prompt = build_control_judge_prompt(video_wm_prompt)

    raw = client.judge(prompt=prompt, video_path=Path(task.wm_video))
    parsed = parse_control_judge_output(raw)

    requested_occlusion = str(parsed.get("requested_occlusion", "")).strip()
    requested_trigger = str(parsed.get("requested_trigger", "")).strip()
    occlusion_done = bool(parsed.get("occlusion_done", False))
    trigger_applied = bool(parsed.get("trigger_applied", False))
    notes = str(parsed.get("notes", "")).strip()

    run_task_dir = Path(task.final_frame).parent  # per_task/<task_id>/
    report_path = run_task_dir / report_filename

    payload = {
        "task_id": task.task_id,
        "provider": "gemini",
        "model": model,
        "wm_video": str(task.wm_video),
        "video_WM_prompt": video_wm_prompt,
        "requested_occlusion": requested_occlusion,
        "requested_trigger": requested_trigger,
        "occlusion_done": occlusion_done,
        "trigger_applied": trigger_applied,
        "notes": notes,
        "raw_text": raw,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return ControlJudgeResult(
        task_id=task.task_id,
        provider="gemini",
        model=model,
        report_path=str(report_path),
        requested_occlusion=requested_occlusion,
        requested_trigger=requested_trigger,
        occlusion_done=occlusion_done,
        trigger_applied=trigger_applied,
        raw_text=raw,
    )


def evaluate_control_all_tasks(
    tasks: Sequence[ResolvedTask],
    *,
    model: str = "gemini-3-pro-preview",
    report_filename: str = "control_report.json",
) -> List[ControlJudgeResult]:
    out: List[ControlJudgeResult] = []
    for t in tasks:
        out.append(
            evaluate_control_one_task(
                t,
                model=model,
                report_filename=report_filename,
            )
        )
    return out


def append_control_results_to_summary(tasks: List[ResolvedTask]) -> None:
    """
    For each task in the same run:
      - Load control_report.json from per-task folder
      - Insert occlusion_done and trigger_applied
        into the corresponding task entry in summary.json

    Assumes all tasks belong to the same run.
    """

    if not tasks:
        return

    # Infer run_dir from first task
    # per_task/<task_id>/... → go up two levels
    first_task_dir = Path(tasks[0].final_frame).parent
    run_dir = first_task_dir.parent.parent
    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if "tasks" not in summary or not isinstance(summary["tasks"], list):
        raise ValueError("summary.json does not contain valid 'tasks' list.")

    # Build quick lookup for task entries
    task_entries = {t["task_id"]: t for t in summary["tasks"] if "task_id" in t}

    for task in tasks:
        task_id = task.task_id
        run_task_dir = Path(task.final_frame).parent
        control_report_path = run_task_dir / "control_report.json"

        if not control_report_path.exists():
            continue  # skip if no control report

        control_data = json.loads(control_report_path.read_text(encoding="utf-8"))

        occlusion_done = bool(control_data.get("occlusion_done", False))
        trigger_applied = bool(control_data.get("trigger_applied", False))

        if task_id not in task_entries:
            continue

        task_entries[task_id]["occlusion_done"] = occlusion_done
        task_entries[task_id]["trigger_applied"] = trigger_applied

    # Write updated summary
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
