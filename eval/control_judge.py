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
    artifact: bool

    raw_text: str


def _load_task_fields(task_yaml_path: Path) -> tuple[str, str, str]:
    """Return (video_WM, camera_WM, camera_pose) from the task YAML. Empty string if absent."""
    data = yaml.safe_load(Path(task_yaml_path).read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts", {}) or {}
    video_wm = prompts.get("video_WM", "") or ""
    camera_wm = prompts.get("camera_WM", "") or ""
    raw_pose = (data.get("camera_control") or {}).get("HY-WorldPlay") or ""
    camera_pose = str(raw_pose).strip() if raw_pose else ""
    return video_wm.strip(), camera_wm.strip(), camera_pose


_OCCLUSION_SIGNAL = "after a pause"  # case-insensitive; present in all occlusion prompts, never in baselines


def _extract_occlusion_clause(video_wm: str) -> str:
    """Extract just the occlusion description from video_WM.

    Handles two prompt shapes:

    1. image_implied (single clause, no semicolon):
         "A curtain drops in front of X, blocking it from view, then, after a pause, rises."
         "A metal lid is placed over Y, covering it from view. After a pause, lifted away."
       → strips the reveal tail; returns the remaining occlusion clause.

    2. simple_kickoff ("<kick-off>; while <process>, <occlusion>. After a pause, <reveal>."):
         "Hand strikes ball; while the balls scatter, the lights turn off, leaving complete
          darkness. After a pause, the lights turn back on."
       → strips the reveal tail, then takes the part after the first ";" —
         the "while <process>, <occlusion>" clause.
    """
    text = video_wm.strip()

    # Strip the reveal tail in both patterns:
    #   A: ", then, after a pause, ..."  (original image_implied style)
    #   B: ". After a pause, ..."        (new object-occlusion / simple_kickoff style)
    text = re.sub(
        r",?\s*then[,\s]+after\s+a\s+pause\b.*", "", text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    text = re.sub(
        r"\.\s*after\s+a\s+pause\b.*", "", text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # For simple_kickoff prompts the kick-off and occlusion are separated by ";".
    # Take only the part after the first ";" — the "while process, occlusion" clause.
    if ";" in text:
        text = text.split(";", 1)[1].strip()

    return text


def _compute_requested_fields(
    video_wm: str,
    camera_wm: str,
    camera_pose: str,
) -> tuple[str, str]:
    """Compute (requested_trigger, requested_occlusion) from task prompt fields.

    requested_trigger:
      - camera_WM if non-empty (simple_kickoff tasks always set camera_WM to the
        kick-off sentence; image_implied tasks always leave it empty)
      - else "none"

    requested_occlusion:
      - occlusion clause extracted from video_WM if it contains the occlusion signal
        ("after a pause"); see _extract_occlusion_clause for the two supported shapes:
          * image_implied: single clause ending in ", then, after a pause, …"
          * simple_kickoff: "<kick-off>; while <process>, <occlusion>. After a pause, …"
      - else "camera pan" if a camera pose string is present
      - else "none" (pure baseline — no occlusion of any kind requested)
    """
    _has_occlusion = _OCCLUSION_SIGNAL in video_wm.lower()

    # --- requested_trigger ---
    # camera_WM is the authoritative source for the kick-off action.
    # image_implied tasks always have camera_WM empty (no kickoff needed).
    # simple_kickoff tasks always have camera_WM filled with the kickoff sentence.
    if camera_wm:
        requested_trigger = camera_wm
    else:
        requested_trigger = "none"

    # --- requested_occlusion ---
    if _has_occlusion:
        requested_occlusion = _extract_occlusion_clause(video_wm)
    elif camera_pose:
        requested_occlusion = "camera pan"
    else:
        requested_occlusion = "none"

    return requested_trigger, requested_occlusion


def build_control_judge_prompt(requested_trigger: str, requested_occlusion: str) -> str:
    trigger_display   = requested_trigger   or "none"
    occlusion_display = requested_occlusion or "none"

    return (
        "You are evaluating a generated video for controllability and visual quality.\n\n"

        "The following have been determined from the task specification:\n"
        f"  requested_trigger:   {trigger_display}\n"
        f"  requested_occlusion: {occlusion_display}\n\n"

        "PARTIAL OBSERVABILITY:\n"
        "The video may hide the scene via in-scene occlusion (lights off, object placed in front,\n"
        "smoke filling the view, etc.) OR by camera pan (camera moves away, taking the subject\n"
        "fully out of frame). Both mechanisms are valid.\n"
        "Do NOT penalise a mismatch between requested_occlusion and the mechanism shown in the video.\n"
        "Occlusion success is judged solely by whether the subject became invisible, regardless of how.\n\n"

        "Evaluate THREE INDEPENDENT things from the GENERATED video:\n\n"

        "1. trigger_applied — Did the requested kickoff action occur?\n"
        "   TRUE if the action itself happened or its physical effect is clearly visible.\n"
        "   Set to true if requested_trigger is \"none\".\n\n"

        "2. occlusion_done — Were the key subject and dynamic successfully hidden from view\n"
        "   (became invisible) at the appropriate moment, by ANY mechanism?\n"
        "   Camera pan that moves the subject fully out of frame counts as occlusion_done = True,\n"
        "   even if requested_occlusion described a different in-scene method — and vice versa.\n"
        "   TRUE only if the main subject/area became clearly invisible or fully obscured.\n"
        "   FALSE if the dynamic process already completed BEFORE the occlusion took place.\n"
        "   FALSE if the scene was never hidden at all.\n"
        "   Set to true if requested_occlusion is \"none\".\n\n"

        "3. artifact — Are there obvious visual artifacts in the video?\n"
        "   TRUE if ANY of the following are clearly visible:\n"
        "     - Unrequested scene changes (background or environment suddenly different)\n"
        "     - Scene reset: the scene briefly blacks out or cuts, then the same objects reappear\n"
        "       in a discontinuous state or different configuration with no physical explanation\n"
        "       (do NOT flag this if requested_occlusion describes lights turning off, since a\n"
        "       brief darkness followed by a scene reveal is expected in that case)\n"
        "     - Object deformation without a physical cause\n"
        "     - Objects appearing or disappearing out of nowhere\n"
        "     - Objects suddenly jumping to a different location with no physical explanation\n"
        "     - Objects morphing into each other or going through each other\n"
        "     - Any other blatantly unrealistic or incoherent visual event\n"
        "   FALSE if the video looks physically plausible throughout.\n\n"

        "DECOUPLING RULE: evaluate trigger_applied, occlusion_done, and artifact independently.\n\n"

        "General guidance:\n"
        "- Use only visual evidence from the video.\n"
        "- Ignore timestamps, watermarks, subtitles, and UI overlays.\n\n"

        "Return ONLY valid JSON in this exact format (no markdown, no commentary):\n"
        "{\n"
        "  \"occlusion_done\": true/false,\n"
        "  \"trigger_applied\": true/false,\n"
        "  \"artifact\": true/false,\n"
        "  \"notes\": \"optional short explanation\"\n"
        "}\n"
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

    video_wm_prompt, camera_wm_prompt, camera_pose = _load_task_fields(Path(task.task_yaml))
    requested_trigger, requested_occlusion = _compute_requested_fields(
        video_wm_prompt, camera_wm_prompt, camera_pose
    )
    prompt = build_control_judge_prompt(requested_trigger, requested_occlusion)

    raw = client.judge(prompt=prompt, video_path=Path(task.wm_video))
    parsed = parse_control_judge_output(raw)

    occlusion_done = bool(parsed.get("occlusion_done", False))
    trigger_applied = bool(parsed.get("trigger_applied", False))
    artifact = bool(parsed.get("artifact", False))
    notes = str(parsed.get("notes", "")).strip()

    run_task_dir = Path(task.final_frame).parent  # per_task/<task_id>/
    report_path = run_task_dir / report_filename

    payload = {
        "task_id": task.task_id,
        "provider": "gemini",
        "model": model,
        "wm_video": str(task.wm_video),
        "video_WM_prompt": video_wm_prompt,
        "camera_WM_prompt": camera_wm_prompt,
        "requested_occlusion": requested_occlusion,
        "requested_trigger": requested_trigger,
        "occlusion_done": occlusion_done,
        "trigger_applied": trigger_applied,
        "artifact": artifact,
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
        artifact=artifact,
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
        artifact = bool(control_data.get("artifact", False))

        if task_id not in task_entries:
            continue

        task_entries[task_id]["occlusion_done"] = occlusion_done
        task_entries[task_id]["trigger_applied"] = trigger_applied
        task_entries[task_id]["artifact"] = artifact

    # Write updated summary
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
