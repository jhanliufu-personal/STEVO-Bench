#!/usr/bin/env python3
# eval/human_eval_server.py
"""
Localhost web UI for human evaluation of world-model benchmark runs.

Dependencies:
    pip install flask

Usage:
    python -m eval.human_eval_server
    python -m eval.human_eval_server --runs_dir runs/ --tasks_dir benchmark/tasks/ --port 7860

Each task page shows:
  - Initial frame
  - Output video (with seeking support)
  - Video WM prompt
  - LLM judge questions + answers
  - Control judge output (occlusion / trigger)
  - T/F buttons for human answers (auto-saved to judge_report.json / control_report.json)
  - Overall LLM and Human verdicts
"""

import argparse
import base64
import json
import mimetypes
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, abort, jsonify, request

try:
    from benchmark.runners.utils import load_yaml
except ImportError:
    import yaml  # type: ignore

    def load_yaml(path):  # type: ignore
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


app = Flask(__name__)

# Set at startup via main()
RUNS_DIR: Path = Path("runs")
TASKS_ROOT: Path = Path("benchmark/tasks")

# Built once at startup: task_id -> yaml path
_YAML_MAP: Dict[str, Path] = {}


def _build_yaml_map() -> None:
    """Scan TASKS_ROOT once and index every <task_id>.yaml file."""
    global _YAML_MAP
    print(f"Scanning tasks directory: {TASKS_ROOT} …", flush=True)
    _YAML_MAP = {p.stem: p for p in TASKS_ROOT.rglob("*.yaml")}
    print(f"  Found {len(_YAML_MAP)} task YAML files.", flush=True)


# ---------------------------------------------------------------------------
# Path encoding helpers
# ---------------------------------------------------------------------------

def _enc(p: str) -> str:
    return base64.urlsafe_b64encode(p.encode("utf-8")).decode("ascii")


def _dec(s: str) -> Path:
    return Path(base64.urlsafe_b64decode(s.encode("ascii")).decode("utf-8"))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _list_runs() -> List[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        d.name for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / "per_task").exists()
    )


def _list_task_ids(run_name: str) -> List[str]:
    pt = RUNS_DIR / run_name / "per_task"
    if not pt.exists():
        return []
    return sorted(
        d.name for d in pt.iterdir()
        if d.is_dir() and (d / "judge_report.json").exists()
    )


def _find_task_yaml(task_id: str) -> Optional[Path]:
    return _YAML_MAP.get(task_id)


def _load_task_data(run_name: str, task_id: str) -> Dict[str, Any]:
    pt_dir = RUNS_DIR / run_name / "per_task" / task_id

    # Judge report
    jr_path = pt_dir / "judge_report.json"
    if not jr_path.exists():
        raise FileNotFoundError(f"judge_report.json missing: {jr_path}")
    jr = json.loads(jr_path.read_text(encoding="utf-8"))

    # Control report
    cr_path = pt_dir / "control_report.json"
    cr: Dict[str, Any] = {}
    if cr_path.exists():
        cr = json.loads(cr_path.read_text(encoding="utf-8"))

    # YAML questions + prompt
    questions: List[Dict[str, str]] = []
    video_wm_prompt: str = cr.get("video_WM_prompt", "") or ""
    yaml_path = _find_task_yaml(task_id)
    if yaml_path:
        td = load_yaml(yaml_path)
        for q in ((td.get("evaluation") or {}).get("questions") or []):
            questions.append({
                "id": str(q.get("id", "")),
                "question": str(q.get("question", "")),
                "notes_for_judge": str(q.get("notes_for_judge", "")),
                "answer_type": str(q.get("answer_type", "yes_no")),
            })
        if not video_wm_prompt:
            video_wm_prompt = (td.get("prompts") or {}).get("video_WM", "") or ""

    # LLM answers indexed by question id
    llm_answers: Dict[str, Any] = {
        a["id"]: {
            "answer":     a.get("answer", ""),
            "confidence": a.get("confidence"),
            "notes":      a.get("notes", ""),
        }
        for a in jr.get("answers", [])
    }

    # Existing human answers (stored as human_answer field inside each answer entry)
    human_answers: Dict[str, str] = {
        a["id"]: a["human_answer"]
        for a in jr.get("answers", [])
        if "human_answer" in a
    }

    # Existing human control answers
    human_control: Dict[str, Optional[bool]] = {}
    if "human_occlusion_done" in cr:
        human_control["occlusion_done"] = cr["human_occlusion_done"]
    if "human_trigger_applied" in cr:
        human_control["trigger_applied"] = cr["human_trigger_applied"]

    def furl(raw: str) -> Optional[str]:
        return f"/file?p={_enc(raw)}" if raw else None

    return {
        "task_id":         task_id,
        "run_name":        run_name,
        "init_frame_url":  furl(jr.get("init_frame", "")),
        "final_frame_url": furl(jr.get("final_frame", "")),
        "video_url":       furl(cr.get("wm_video", "")),
        "video_wm_prompt": video_wm_prompt,
        "questions":       questions,
        "llm_answers":     llm_answers,
        "llm_control": {
            "requested_occlusion": cr.get("requested_occlusion", ""),
            "requested_trigger":   cr.get("requested_trigger", ""),
            "occlusion_done":      cr.get("occlusion_done"),
            "trigger_applied":     cr.get("trigger_applied"),
            "notes":               cr.get("notes", ""),
        },
        "human_answers": human_answers,
        "human_control": human_control,
        "score":         jr.get("score", {}),
    }


# ---------------------------------------------------------------------------
# Range-aware file serving (required for video seeking in the browser)
# ---------------------------------------------------------------------------

def _serve_range(path: Path) -> Response:
    file_size = path.stat().st_size
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    rng = request.headers.get("Range", "")

    if not rng or not rng.startswith("bytes="):
        with open(path, "rb") as f:
            data = f.read()
        r = Response(data, status=200, mimetype=mime)
        r.headers["Accept-Ranges"] = "bytes"
        r.headers["Content-Length"] = str(file_size)
        return r

    spec = rng[6:].strip()
    start_s, _, end_s = spec.partition("-")
    start  = int(start_s) if start_s else 0
    end    = int(end_s)   if end_s   else file_size - 1
    end    = min(end, file_size - 1)
    length = end - start + 1

    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(length)

    r = Response(data, status=206, mimetype=mime)
    r.headers["Content-Range"]  = f"bytes {start}-{end}/{file_size}"
    r.headers["Content-Length"] = str(length)
    r.headers["Accept-Ranges"]  = "bytes"
    return r


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return Response(_HTML, status=200, mimetype="text/html; charset=utf-8")


@app.route("/api/runs")
def api_runs():
    return jsonify(_list_runs())


@app.route("/api/runs/<run_name>/tasks")
def api_tasks(run_name: str):
    return jsonify(_list_task_ids(run_name))


@app.route("/api/runs/<run_name>/random")
def api_random(run_name: str):
    ids = _list_task_ids(run_name)
    if not ids:
        abort(404, "No tasks found in run")
    return jsonify({"task_id": random.choice(ids), "total": len(ids)})


@app.route("/api/runs/<run_name>/task/<task_id>")
def api_task(run_name: str, task_id: str):
    try:
        return jsonify(_load_task_data(run_name, task_id))
    except FileNotFoundError as e:
        abort(404, str(e))
    except Exception as e:
        abort(500, str(e))


@app.route("/api/runs/<run_name>/task/<task_id>/answer", methods=["POST"])
def api_answer(run_name: str, task_id: str):
    body = request.get_json(force=True) or {}
    pt_dir = RUNS_DIR / run_name / "per_task" / task_id

    # Update judge_report.json — add/overwrite human_answer in each answer entry
    jr_path = pt_dir / "judge_report.json"
    if jr_path.exists():
        jr = json.loads(jr_path.read_text(encoding="utf-8"))
        human = body.get("answers", {})
        for a in jr.get("answers", []):
            if a["id"] in human:
                a["human_answer"] = human[a["id"]]
        jr_path.write_text(json.dumps(jr, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update control_report.json
    cr_path = pt_dir / "control_report.json"
    ctrl = body.get("control", {})
    if cr_path.exists() and ctrl:
        cr = json.loads(cr_path.read_text(encoding="utf-8"))
        if "occlusion_done" in ctrl:
            cr["human_occlusion_done"] = ctrl["occlusion_done"]
        if "trigger_applied" in ctrl:
            cr["human_trigger_applied"] = ctrl["trigger_applied"]
        cr_path.write_text(json.dumps(cr, indent=2, ensure_ascii=False), encoding="utf-8")

    return jsonify({"status": "ok"})


@app.route("/file")
def serve_file():
    p = request.args.get("p", "")
    if not p:
        abort(400, "Missing p parameter")
    try:
        path = _dec(p)
    except Exception:
        abort(400, "Bad path encoding")
    if not path.exists():
        abort(404, "File not found")
    try:
        return _serve_range(path)
    except Exception as e:
        abort(500, str(e))


# ---------------------------------------------------------------------------
# Inline HTML / CSS / JS
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>StateWM Human Eval</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --bg:          #0d1117;
      --card-bg:     #161b22;
      --card-head:   #21262d;
      --border:      #30363d;
      --text:        #c9d1d9;
      --muted:       #8b949e;
      --accent:      #58a6ff;
      --green:       #3fb950;
      --red:         #f85149;
    }
    * { box-sizing: border-box; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; min-height: 100vh; }

    /* ── Top bar ── */
    #topbar {
      background: var(--card-bg); border-bottom: 1px solid var(--border);
      padding: 8px 18px; display: flex; align-items: center; gap: 10px;
      position: sticky; top: 0; z-index: 100; flex-wrap: wrap;
    }
    #topbar .brand { font-weight: 700; font-size: 15px; color: var(--accent); white-space: nowrap; }
    #topbar select, #topbar input[type=text] {
      background: var(--card-head); color: var(--text);
      border: 1px solid var(--border); border-radius: 6px;
      padding: 5px 10px; font-size: 13px;
    }
    #topbar input[type=text] { width: 240px; }
    #task-search-wrap { position: relative; }
    #search-dropdown {
      position: absolute; top: calc(100% + 4px); left: 0; min-width: 260px;
      background: var(--card-head); border: 1px solid var(--border);
      border-radius: 6px; max-height: 220px; overflow-y: auto; z-index: 200; display: none;
    }
    .search-item { padding: 7px 12px; cursor: pointer; font-size: 13px; }
    .search-item:hover { background: var(--border); }
    #task-info { color: var(--muted); font-size: 12px; white-space: nowrap; }
    #task-info .tid { color: var(--accent); font-weight: 600; }
    .spacer { flex: 1; }
    .save-chip {
      font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 10px;
      display: none;
    }
    .save-chip.saving { background: #d29922; color: #000; display: inline-block; }
    .save-chip.saved  { background: var(--green); color: #000; display: inline-block; }
    .save-chip.error  { background: var(--red);   color: #fff; display: inline-block; }

    /* ── Buttons ── */
    .btn-primary-sm {
      background: var(--accent); color: #0d1117; border: none;
      padding: 6px 16px; border-radius: 6px; font-weight: 600; font-size: 13px; cursor: pointer;
    }
    .btn-primary-sm:hover { background: #79c0ff; }
    .btn-primary-sm:disabled { opacity: 0.45; cursor: default; }

    /* ── Card ── */
    .ev-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 14px; overflow: hidden; }
    .ev-card-header { background: var(--card-head); padding: 7px 14px; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--accent); border-bottom: 1px solid var(--border); }
    .ev-card-body { padding: 10px; }

    /* ── Verdict row ── */
    .verdict-row { display: flex; gap: 10px; margin-bottom: 14px; }
    .v-box {
      flex: 1; text-align: center; padding: 12px 8px; border-radius: 8px;
      font-size: 15px; font-weight: 700; border: 2px solid var(--border);
      background: var(--card-head); color: var(--muted);
    }
    .v-pass { background: rgba(63,185,80,0.12); border-color: var(--green); color: var(--green); }
    .v-fail { background: rgba(248,81,73,0.12);  border-color: var(--red);   color: var(--red); }
    .v-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 400; margin-bottom: 2px; }

    /* ── Answer badge ── */
    .abadge {
      display: inline-block; padding: 2px 9px; border-radius: 10px;
      font-size: 11px; font-weight: 600; text-transform: uppercase;
    }
    .ab-yes { background: rgba(63,185,80,0.18); color: var(--green); border: 1px solid var(--green); }
    .ab-no  { background: rgba(248,81,73,0.18); color: var(--red);   border: 1px solid var(--red); }
    .ab-unk { background: rgba(139,148,158,0.15); color: var(--muted); border: 1px solid var(--border); }

    /* ── T / F toggle buttons ── */
    .tf-wrap { display: flex; gap: 4px; justify-content: center; }
    .btn-tf {
      padding: 3px 13px; font-size: 12px; font-weight: 700; border-radius: 5px;
      border: 1.5px solid var(--border); background: transparent;
      color: var(--muted); cursor: pointer; transition: all 0.12s;
    }
    .btn-tf:hover { background: var(--card-head); }
    .btn-tf.on-t  { background: rgba(63,185,80,0.2); border-color: var(--green); color: var(--green); }
    .btn-tf.on-f  { background: rgba(248,81,73,0.2); border-color: var(--red);   color: var(--red); }

    /* ── Question table ── */
    .q-table { width: 100%; border-collapse: separate; border-spacing: 0 5px; }
    .q-table th {
      color: var(--muted); font-size: 11px; text-transform: uppercase;
      letter-spacing: 0.5px; padding: 4px 8px; font-weight: 600;
    }
    .q-row td {
      background: var(--card-head); padding: 9px 10px;
      border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
      vertical-align: middle;
    }
    .q-row td:first-child { border-left: 1px solid var(--border); border-radius: 6px 0 0 6px; }
    .q-row td:last-child  { border-right: 1px solid var(--border); border-radius: 0 6px 6px 0; }
    .q-text  { font-size: 13px; line-height: 1.4; }
    .q-notes { font-size: 11px; color: var(--muted); margin-top: 3px; font-style: italic; }
    .conf-pct { font-size: 10px; color: var(--muted); margin-top: 2px; }

    /* ── Control rows ── */
    .ctrl-row {
      display: flex; align-items: flex-start; gap: 10px; padding: 10px;
      background: var(--card-head); border-radius: 6px; margin-bottom: 8px;
      border: 1px solid var(--border);
    }
    .ctrl-body { flex: 1; }
    .ctrl-lbl  { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 3px; }
    .ctrl-val  { font-size: 13px; }
    .ctrl-llm  { display: flex; flex-direction: column; align-items: center; gap: 3px; min-width: 54px; }
    .ctrl-human { display: flex; flex-direction: column; align-items: center; gap: 3px; min-width: 66px; }
    .ctrl-notes {
      font-size: 12px; color: var(--muted); font-style: italic;
      padding: 7px 10px; background: var(--bg); border-radius: 5px;
      border-left: 3px solid var(--border); margin-top: 4px;
    }

    /* ── Media ── */
    #init-frame   { width: 100%; max-height: 340px; object-fit: contain; border-radius: 5px; background: #000; display: block; }
    #output-video { width: 100%; border-radius: 5px; background: #000; display: block; }
    .media-ph {
      width: 100%; height: 180px; background: var(--card-head); border-radius: 5px;
      border: 1px dashed var(--border); display: flex; align-items: center;
      justify-content: center; color: var(--muted); font-size: 13px;
    }
    .prompt-box {
      background: var(--card-head); border: 1px solid var(--border); border-radius: 6px;
      padding: 11px; font-size: 13px; line-height: 1.6; white-space: pre-wrap;
      color: var(--text); max-height: 200px; overflow-y: auto;
    }

    /* ── Misc ── */
    #empty-state { text-align: center; padding: 80px 20px; color: var(--muted); }
    #empty-state .big { font-size: 52px; }
    #empty-state .msg { font-size: 20px; margin-top: 10px; }
    #loading-overlay {
      position: fixed; inset: 0; background: rgba(0,0,0,0.45);
      display: none; align-items: center; justify-content: center; z-index: 999;
    }
    .spinner {
      width: 42px; height: 42px; border: 4px solid var(--border);
      border-top-color: var(--accent); border-radius: 50%;
      animation: spin 0.75s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  </style>
</head>
<body>

<!-- Loading overlay -->
<div id="loading-overlay"><div class="spinner"></div></div>

<!-- Top bar -->
<div id="topbar">
  <span class="brand">StateWM Eval</span>

  <select id="run-select">
    <option value="">— Select run —</option>
  </select>

  <div id="task-search-wrap">
    <input type="text" id="task-search" placeholder="Search task ID…" autocomplete="off" disabled>
    <div id="search-dropdown"></div>
  </div>

  <button class="btn-primary-sm" id="btn-random" disabled>→ Random task</button>

  <span class="spacer"></span>

  <span id="task-info" style="display:none">
    <span class="tid" id="info-tid"></span>
    &nbsp;·&nbsp;
    <span id="info-remaining"></span>
  </span>

  <span id="save-chip" class="save-chip"></span>
</div>

<!-- Empty state -->
<div id="empty-state">
  <div class="big">🔍</div>
  <div class="msg">Select a run to begin</div>
</div>

<!-- Task view -->
<div id="task-view" style="display:none; padding: 16px 20px; max-width: 1400px; margin: 0 auto;">
  <div class="row g-3">

    <!-- Left column: media -->
    <div class="col-lg-5">

      <div class="ev-card">
        <div class="ev-card-header">Initial Frame</div>
        <div class="ev-card-body">
          <img id="init-frame" alt="Initial frame" style="display:none">
          <div id="init-ph" class="media-ph">No image available</div>
        </div>
      </div>

      <div class="ev-card">
        <div class="ev-card-header">Output Video</div>
        <div class="ev-card-body">
          <video id="output-video" controls style="display:none"></video>
          <div id="video-ph" class="media-ph">No video available</div>
        </div>
      </div>

      <div class="ev-card">
        <div class="ev-card-header">Video WM Prompt</div>
        <div class="ev-card-body">
          <div id="video-prompt" class="prompt-box"></div>
        </div>
      </div>

    </div><!-- /left -->

    <!-- Right column: evaluation -->
    <div class="col-lg-7">

      <!-- Verdicts -->
      <div class="verdict-row">
        <div class="v-box" id="llm-verdict">
          <div class="v-label">LLM Verdict</div>
          <span>—</span>
        </div>
        <div class="v-box" id="human-verdict">
          <div class="v-label">Human Verdict</div>
          <span>—</span>
        </div>
      </div>

      <!-- Judge questions -->
      <div class="ev-card">
        <div class="ev-card-header">Judge Questions</div>
        <div class="ev-card-body" style="overflow-x:auto;">
          <table class="q-table">
            <thead>
              <tr>
                <th style="width:48%">Question</th>
                <th style="width:22%; text-align:center">LLM Answer</th>
                <th style="width:30%; text-align:center">Human</th>
              </tr>
            </thead>
            <tbody id="q-body"></tbody>
          </table>
        </div>
      </div>

      <!-- Control judge -->
      <div class="ev-card">
        <div class="ev-card-header">Control Judge</div>
        <div class="ev-card-body" id="ctrl-body">
          <div style="color:var(--muted); font-size:13px;">No control data</div>
        </div>
      </div>

    </div><!-- /right -->
  </div>
</div>

<script>
// =============================================================================
// State
// =============================================================================
let S = {
  run:          null,   // string
  taskId:       null,   // string
  questions:    [],     // [{id, question, notes_for_judge, answer_type}]
  llmAnswers:   {},     // {id: {answer, confidence, notes}}
  llmControl:   {},     // {requested_occlusion, requested_trigger, occlusion_done, trigger_applied, notes}
  humanAnswers: {},     // {id: "yes"|"no"}  — current human answers
  humanControl: {},     // {occlusion_done: bool, trigger_applied: bool}
  allTaskIds:   [],     // for search
};

let _saveTimer = null;

// =============================================================================
// Initialise
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
  loadRuns();

  document.getElementById('run-select').addEventListener('change', e => {
    if (e.target.value) selectRun(e.target.value);
  });
  document.getElementById('btn-random').addEventListener('click', loadRandomTask);

  // Task search
  const inp = document.getElementById('task-search');
  const dd  = document.getElementById('search-dropdown');
  inp.addEventListener('input', () => {
    const q = inp.value.trim().toLowerCase();
    if (!q || !S.allTaskIds.length) { dd.style.display = 'none'; return; }
    const hits = S.allTaskIds.filter(id => id.toLowerCase().includes(q)).slice(0, 25);
    if (!hits.length) { dd.style.display = 'none'; return; }
    dd.innerHTML = hits.map(id =>
      `<div class="search-item" onclick="pickTask('${esc(id)}')">${esc(id)}</div>`
    ).join('');
    dd.style.display = 'block';
  });
  document.addEventListener('click', e => {
    if (!document.getElementById('task-search-wrap').contains(e.target))
      dd.style.display = 'none';
  });
});

// =============================================================================
// API helpers
// =============================================================================
async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

// =============================================================================
// Run selection
// =============================================================================
async function loadRuns() {
  const runs = await api('/api/runs');
  const sel = document.getElementById('run-select');
  sel.innerHTML = '<option value="">— Select run —</option>' +
    runs.map(r => `<option value="${esc(r)}">${esc(r)}</option>`).join('');
}

async function selectRun(runName) {
  S.run = runName;
  setLoading(true);
  try {
    S.allTaskIds = await api(`/api/runs/${enc(runName)}/tasks`);
    document.getElementById('task-search').disabled = false;
    document.getElementById('btn-random').disabled  = false;
    await loadRandomTask();
  } catch(e) {
    alert('Failed to load run: ' + e.message);
  } finally {
    setLoading(false);
  }
}

// =============================================================================
// Task navigation
// =============================================================================
async function loadRandomTask() {
  if (!S.run) return;
  setLoading(true);
  try {
    const { task_id, total } = await api(`/api/runs/${enc(S.run)}/random`);
    updateInfoBar(task_id, total);
    await loadTask(task_id);
  } catch(e) {
    alert('Failed to load random task: ' + e.message);
  } finally {
    setLoading(false);
  }
}

async function pickTask(taskId) {
  document.getElementById('search-dropdown').style.display = 'none';
  document.getElementById('task-search').value = taskId;
  setLoading(true);
  try {
    await loadTask(taskId);
  } finally {
    setLoading(false);
  }
}

async function loadTask(taskId) {
  const data = await api(`/api/runs/${enc(S.run)}/task/${enc(taskId)}`);
  S.taskId       = taskId;
  S.questions    = data.questions   || [];
  S.llmAnswers   = data.llm_answers || {};
  S.llmControl   = data.llm_control || {};
  S.humanAnswers = { ...data.human_answers };   // copy so we can mutate
  S.humanControl = { ...data.human_control };
  updateInfoBar(taskId, null);
  render(data);
  hideSave();
}

function updateInfoBar(taskId, total) {
  document.getElementById('info-tid').textContent = taskId;
  if (total != null)
    document.getElementById('info-remaining').textContent = `${total} tasks`;
  document.getElementById('task-info').style.display = '';
}

// =============================================================================
// Rendering
// =============================================================================
function render(data) {
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('task-view').style.display   = '';

  // Init frame
  const img = document.getElementById('init-frame');
  const iph = document.getElementById('init-ph');
  if (data.init_frame_url) {
    img.src = data.init_frame_url; img.style.display = ''; iph.style.display = 'none';
  } else {
    img.style.display = 'none'; iph.style.display = '';
  }

  // Video
  const vid = document.getElementById('output-video');
  const vph = document.getElementById('video-ph');
  if (data.video_url) {
    vid.src = data.video_url; vid.load(); vid.style.display = ''; vph.style.display = 'none';
  } else {
    vid.style.display = 'none'; vph.style.display = '';
  }

  // Prompt
  document.getElementById('video-prompt').textContent = data.video_wm_prompt || '(none)';

  // Questions
  renderQuestions();

  // Control
  renderControl();

  // Verdicts
  refreshVerdicts();
}

function renderQuestions() {
  const tbody = document.getElementById('q-body');
  if (!S.questions.length) {
    tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--muted);padding:14px">No questions</td></tr>';
    return;
  }
  tbody.innerHTML = S.questions.map(q => {
    const llm  = S.llmAnswers[q.id] || {};
    const hAns = S.humanAnswers[q.id];
    return `<tr class="q-row">
      <td>
        <div class="q-text">${esc(q.question)}</div>
        ${q.notes_for_judge ? `<div class="q-notes">${esc(q.notes_for_judge)}</div>` : ''}
      </td>
      <td style="text-align:center">
        ${ansBadge(llm.answer)}
        ${llm.confidence != null ? `<div class="conf-pct">${Math.round(llm.confidence*100)}%</div>` : ''}
        ${llm.notes ? `<div class="q-notes" style="max-width:160px">${esc(llm.notes)}</div>` : ''}
      </td>
      <td>
        <div class="tf-wrap">
          <button class="btn-tf ${hAns==='yes'?'on-t':''}" onclick="setAnswer('${esc(q.id)}','yes')">T</button>
          <button class="btn-tf ${hAns==='no' ?'on-f':''}" onclick="setAnswer('${esc(q.id)}','no')">F</button>
        </div>
      </td>
    </tr>`;
  }).join('');
}

function renderControl() {
  const body = document.getElementById('ctrl-body');
  const c = S.llmControl;
  if (c.occlusion_done == null && c.trigger_applied == null) {
    body.innerHTML = '<div style="color:var(--muted);font-size:13px;">No control data</div>';
    return;
  }
  const hOcc  = S.humanControl.occlusion_done;
  const hTrig = S.humanControl.trigger_applied;

  body.innerHTML = `
    <div class="ctrl-row">
      <div class="ctrl-body">
        <div class="ctrl-lbl">Occlusion method</div>
        <div class="ctrl-val">${esc(c.requested_occlusion || '—')}</div>
      </div>
      <div class="ctrl-llm">
        <div class="ctrl-lbl">LLM</div>
        ${boolBadge(c.occlusion_done)}
      </div>
      <div class="ctrl-human">
        <div class="ctrl-lbl">Human</div>
        <div class="tf-wrap">
          <button class="btn-tf ${hOcc===true ?'on-t':''}" onclick="setControl('occlusion_done',true)">T</button>
          <button class="btn-tf ${hOcc===false?'on-f':''}" onclick="setControl('occlusion_done',false)">F</button>
        </div>
      </div>
    </div>

    <div class="ctrl-row">
      <div class="ctrl-body">
        <div class="ctrl-lbl">Trigger / action</div>
        <div class="ctrl-val">${esc(c.requested_trigger || '—')}</div>
      </div>
      <div class="ctrl-llm">
        <div class="ctrl-lbl">LLM</div>
        ${boolBadge(c.trigger_applied)}
      </div>
      <div class="ctrl-human">
        <div class="ctrl-lbl">Human</div>
        <div class="tf-wrap">
          <button class="btn-tf ${hTrig===true ?'on-t':''}" onclick="setControl('trigger_applied',true)">T</button>
          <button class="btn-tf ${hTrig===false?'on-f':''}" onclick="setControl('trigger_applied',false)">F</button>
        </div>
      </div>
    </div>

    ${c.notes ? `<div class="ctrl-notes">${esc(c.notes)}</div>` : ''}
  `;
}

// =============================================================================
// Human answer handlers
// =============================================================================
function setAnswer(qid, value) {
  S.humanAnswers[qid] = value;

  // Update button styles without full re-render
  const rows = document.querySelectorAll('#q-body .q-row');
  rows.forEach(row => {
    const btnT = row.querySelector('.btn-tf:first-child');
    if (!btnT) return;
    // Find the question id from onclick attribute
    const onclickT = btnT.getAttribute('onclick') || '';
    const m = onclickT.match(/setAnswer\('([^']+)','yes'\)/);
    if (!m || m[1] !== qid) return;
    const btnF = row.querySelector('.btn-tf:last-child');
    btnT.className = 'btn-tf' + (value === 'yes' ? ' on-t' : '');
    btnF.className = 'btn-tf' + (value === 'no'  ? ' on-f' : '');
  });

  refreshVerdicts();
  scheduleSave();
}

function setControl(field, value) {
  S.humanControl[field] = value;

  // Update button styles
  const suffix = field === 'occlusion_done' ? 'occlusion_done' : 'trigger_applied';
  document.querySelectorAll(`[onclick*="setControl('${field}"]`).forEach(btn => {
    const isT = btn.getAttribute('onclick').includes(',true)');
    btn.className = 'btn-tf' + (
      (isT && value === true)  ? ' on-t' :
      (!isT && value === false) ? ' on-f' : ''
    );
  });

  scheduleSave();
}

// =============================================================================
// Verdicts
// =============================================================================
function refreshVerdicts() {
  const qids = S.questions.map(q => q.id);

  // LLM: fail if any answer == "no"
  const llmHasNo  = qids.some(id => (S.llmAnswers[id]?.answer || '').toLowerCase() === 'no');
  const llmHasAny = qids.some(id => !!S.llmAnswers[id]?.answer);
  setVerdict('llm-verdict', 'LLM',
    !llmHasAny ? 'unk' : llmHasNo ? 'fail' : 'pass');

  // Human: fail if any answer == "no"; unknown if not all answered
  const answered = qids.filter(id => S.humanAnswers[id] !== undefined);
  const humanFail = answered.some(id => S.humanAnswers[id] === 'no');
  const allAnswered = answered.length === qids.length && qids.length > 0;
  setVerdict('human-verdict', 'Human',
    answered.length === 0 ? 'unk' : humanFail ? 'fail' : allAnswered ? 'pass' : 'partial');
}

function setVerdict(id, label, state) {
  const el = document.getElementById(id);
  const labels = { 'LLM': 'LLM Verdict', 'Human': 'Human Verdict' };
  const content = {
    pass:    `<div class="v-label">${labels[label]}</div><span>PASS ✓</span>`,
    fail:    `<div class="v-label">${labels[label]}</div><span>FAIL ✗</span>`,
    partial: `<div class="v-label">${labels[label]}</div><span>PARTIAL</span>`,
    unk:     `<div class="v-label">${labels[label]}</div><span>—</span>`,
  };
  el.innerHTML = content[state] || content.unk;
  el.className = 'v-box' + (state === 'pass' ? ' v-pass' : state === 'fail' ? ' v-fail' : '');
}

// =============================================================================
// Save (debounced)
// =============================================================================
function scheduleSave() {
  clearTimeout(_saveTimer);
  showSave('saving');
  _saveTimer = setTimeout(doSave, 450);
}

async function doSave() {
  if (!S.run || !S.taskId) return;
  try {
    await api(
      `/api/runs/${enc(S.run)}/task/${enc(S.taskId)}/answer`,
      {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ answers: S.humanAnswers, control: S.humanControl }),
      }
    );
    showSave('saved');
    setTimeout(hideSave, 2500);
  } catch(e) {
    console.error('Save failed:', e);
    showSave('error');
  }
}

// =============================================================================
// Badge helpers
// =============================================================================
function ansBadge(ans) {
  if (!ans) return '<span class="abadge ab-unk">?</span>';
  const cls = ans === 'yes' ? 'ab-yes' : ans === 'no' ? 'ab-no' : 'ab-unk';
  return `<span class="abadge ${cls}">${esc(ans)}</span>`;
}

function boolBadge(val) {
  if (val === true)  return '<span class="abadge ab-yes">T</span>';
  if (val === false) return '<span class="abadge ab-no">F</span>';
  return '<span class="abadge ab-unk">?</span>';
}

// =============================================================================
// UI utilities
// =============================================================================
function showSave(state) {
  const el = document.getElementById('save-chip');
  el.className = 'save-chip ' + state;
  el.textContent = state === 'saving' ? 'Saving…' : state === 'saved' ? 'Saved ✓' : 'Save failed!';
}
function hideSave() {
  document.getElementById('save-chip').className = 'save-chip';
}
function setLoading(on) {
  document.getElementById('loading-overlay').style.display = on ? 'flex' : 'none';
}
function enc(s) { return encodeURIComponent(s); }
function esc(s) {
  return String(s || '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global RUNS_DIR, TASKS_ROOT

    parser = argparse.ArgumentParser(description="StateWM Human Eval UI")
    parser.add_argument("--runs_dir",  default="runs",            help="Path to runs directory (default: runs/)")
    parser.add_argument("--tasks_dir", default="benchmark/tasks", help="Path to benchmark tasks root (default: benchmark/tasks/)")
    parser.add_argument("--host",      default="127.0.0.1",       help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port",      default=7860, type=int,    help="Port to listen on (default: 7860)")
    args = parser.parse_args()

    RUNS_DIR   = Path(args.runs_dir).expanduser().resolve()
    TASKS_ROOT = Path(args.tasks_dir).expanduser().resolve()

    print("StateWM Human Eval UI")
    print(f"  Runs dir  : {RUNS_DIR}")
    print(f"  Tasks dir : {TASKS_ROOT}")
    print()

    _build_yaml_map()

    print(f"  URL       : http://{args.host}:{args.port}")
    print()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
