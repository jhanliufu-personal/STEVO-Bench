import json
import re
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass

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
    task_level: int
    provider: str
    model: str

    init_frame: str
    final_frame: str

    raw_text: str
    answers: List[JudgeAnswer]

    report_path: str

def _normalize_answer(s: str) -> AnswerValue:
    s = s.strip().lower()
    if s in ("yes", "y", "true"):
        return "yes"
    if s in ("no", "n", "false"):
        return "no"
    return "unknown"

def _unwrap_full_fence(s: str) -> str:
    """
    If entire string is a ```...``` fenced block, unwrap it.
    Otherwise return unchanged.
    """
    s = (s or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s

def _extract_fenced_block(s: str) -> Optional[str]:
    """
    Extract the FIRST fenced block content if present (not necessarily entire string).
    """
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def _extract_first_object(s: str) -> Optional[str]:
    """
    Extract the first {...} JSON object from text.
    """
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0).strip() if m else None

def _try_load_json(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def parse_judge_output(raw_text: str, expected_ids: List[str]) -> List[JudgeAnswer]:
    raw = (raw_text or "").strip()

    answers: Dict[str, JudgeAnswer] = {}

    # 1) If the whole thing is fenced, unwrap and try JSON
    s0 = _unwrap_full_fence(raw)
    data = _try_load_json(s0)

    # 2) If not JSON, try parsing raw directly (covers pure JSON w/o fences)
    if data is None:
        data = _try_load_json(raw)

    # 3) If still not JSON, try JSON inside a fenced block anywhere
    if data is None:
        fenced = _extract_fenced_block(raw)
        if fenced:
            fenced2 = _unwrap_full_fence(fenced)  # harmless if not fully fenced
            data = _try_load_json(fenced2)
            if data is None:
                obj = _extract_first_object(fenced2)
                if obj:
                    data = _try_load_json(obj)

    # 4) If still not JSON, extract first {...} anywhere (handles reasoning before JSON)
    if data is None:
        obj = _extract_first_object(raw)
        if obj:
            data = _try_load_json(obj)

    # Parse JSON answers if available
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

    # 5) Fallback: parse lines like "q01: yes"
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

    # Fill missing with unknown in expected order
    out: List[JudgeAnswer] = []
    for qid in expected_ids:
        out.append(answers.get(qid, JudgeAnswer(id=qid, answer="unknown")))

    return out
