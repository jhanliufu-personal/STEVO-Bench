#!/usr/bin/env python3
"""
Generate task variants along two axes:
1. Occlusion method (camera rotation -> object blocking, zoom out, etc.)
2. Dynamic conditions (key variables that affect state evolution)

IMPORTANT: Always creates a fully observable baseline variant (v0) first.

Usage:
  python benchmark/runners/generate_task_variants.py \
    --level_dir benchmark/tasks/level1_scalar_state \
    --model gemini-3-pro-preview \
    --temperature 0.7 \
    --num_variants 8

Env:
  export GOOGLE_API_KEY="..."
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from google import genai

from utils import load_yaml, dump_yaml, resolve_task_paths, image_to_inline_data, extract_json

# -----------------------------
# Master prompt for variant generation
# -----------------------------

MASTER_PROMPT = """
# StateWMBench: Task Variant Generation

## Benchmark Overview

StateWMBench tests whether video-based world models can accurately maintain and evolve hidden state under **partial observability**. The benchmark evaluates models' ability to:

1. **Track occluded objects** (object permanence)
2. **Infer continuous state changes** based on implied physical conditions
3. **Simulate dynamics** when the scene is temporarily out of view
4. **Reason about causal mechanisms** and relational dependencies

## Task Structure & Design Principles

Each task consists of:

### 1. Initial Scene Setup (initial frame image)
The starting visual state with all relevant objects, properties, and conditions.

### 2. Video Prompt (`video_WM`)
**Two-part structure**:

**Part A - Scene Description & Implied Dynamics**:
- Describes physical actions and conditions
- **IMPLIES but does NOT explicitly state** the resulting physics
- Example: "A cooktop displaying 350F" -> implies heat, but doesn't say "the ice will melt"
- Example: "A smooth, frictionless ramp" -> implies sliding, but doesn't say "the block will accelerate"

**Part B - Camera Movement (Creates Occlusion)**:
- Camera moves to occlude the scene
- Critical state evolution happens OFF-SCREEN
- Camera returns to reveal final state
- Standard template: "The camera rotates right until [object] completely exits the view, then rotates toward the left smoothly at the same rate until [location] is back in the center of view. No zoom, no cut, no scene change. STOP."

### 3. Camera WM Prompt (`camera_WM`)
`video_WM` with camera-based movement instructions removed. For camera-based occlusion
(panning, rotation, zoom), strip those camera movement sentences while keeping everything
else. For non-camera occlusion (lights off, fog, blocking objects), `camera_WM` is
identical to `video_WM` — those are scene events, not camera instructions.

### Key Design Principles:
- **Implied physics**: Conditions suggest dynamics without spelling them out
- **Partial observability**: State evolves while occluded
- **No cuts**: Continuous temporal reasoning required
- **Directional evaluation**: Tests whether state changed in the correct direction, not specific endpoints

## Two Axes of Variation

Your task is to generate variants along TWO independent axes:

---

### AXIS 1: Occlusion Method

**Original method**: Camera rotation (pans away and back)

**Alternative occlusion methods**:

1. **Object blocking**: A large opaque object (curtain, screen, door, person) moves in front of the camera, blocking the view completely, then moves away
2. **Camera zoom out**: Camera zooms out until the scene becomes too small to see details, then zooms back in
3. **Lighting change**: Lights turn off completely (pitch black), then turn back on
4. **Fog/smoke**: Thick fog or smoke fills the space, obscuring the scene, then clears
5. **Camera moves away**: Camera physically moves backward/away from the scene, then returns
6. **Occlusion by foreground**: An object in the foreground moves to block the line of sight, then moves aside

**Requirements**:
- Occlusion must be COMPLETE (scene fully hidden)
- Same duration of occlusion as original
- Must maintain "no cut, no scene change" principle
- Occlusion method should feel natural for the scene

---

### AXIS 2: Dynamic Conditions

**Original task** has implicit conditions that determine whether/how state evolves.

**Task**: Identify key variables and create variants that change these conditions.

**Examples by category**:

**For kinematic tasks** (projectiles, sliding, pendulums):
- Surface friction (frictionless -> high friction)
- Initial velocity (slow -> fast)
- Mass (light -> heavy)
- Angle of incline (steep -> shallow)
- Air resistance (vacuum -> normal air)

**For thermal tasks** (melting, heating, cooling):
- Temperature differential (hot surface -> lukewarm)
- Initial object state (frozen solid -> partially thawed)
- Thermal conductivity (metal surface -> wood surface)
- Ambient conditions (hot room -> cold room)

**For phase change tasks** (inflating, deflating, dissolving):
- Rate of change (fast pump -> slow leak)
- Initial state (empty -> partially filled)
- Pressure differential (high -> low)
- Concentration (pure -> diluted)

**For relational tasks** (pulleys, collisions, reflections):
- Mass ratios (equal masses -> unequal)
- Coupling strength (tight -> loose)
- Symmetry (symmetric -> asymmetric)
- Initial energy (high -> low)

**Requirements**:
- Variants should test the MODEL'S understanding of how conditions affect dynamics
- Some variants should have NO state change (e.g., ice on cold surface -> no melting)
- Some variants should have OPPOSITE direction of change (e.g., cooling instead of heating)
- Some variants should have FASTER/SLOWER rates of change
- Conditions should be IMPLIED in the scene description, not explicitly stated

---

## Your Task

You are given an ORIGINAL task with:
- `id`: Task identifier
- `level`: Complexity level (0-5)
- `category`: Task category
- `image_gen_prompt`: Initial scene description
- `video_WM`: Full video generation prompt
- `camera_WM`: Simplified camera prompt

**Generate {num_variants} task variants** that explore both axes systematically.

### Variant Distribution:
- **{occlusion_variants} variants**: Different occlusion methods (AXIS 1)
- **{condition_variants} variants**: Different dynamic conditions (AXIS 2)
- **{combined_variants} variants**: Combination of both axes

### For Each Variant, Provide:

```json
{{
  "variant_id": "original_task_id_02",
  "variant_suffix": "blocking",
  "variation_type": "occlusion_method|dynamic_condition|combined",
  "variation_description": "Brief description of what changed",
  "changes": {{
    "occlusion_method": "Description of new occlusion method (or null if unchanged)",
    "dynamic_conditions": "Description of changed conditions (or null if unchanged)",
    "expected_outcome_change": "How the final state differs from original (or 'same dynamics' if only occlusion changed)"
  }},
  "init_frame_edit_prompt": "CRITICAL: You are given the ORIGINAL initial frame image. If dynamic conditions changed, provide an image EDIT prompt starting with 'Edit the image to...'. If ONLY occlusion method changed, return '[NO CHANGE]' to reuse the original frame unchanged.",
  "video_WM": "Modified full video prompt. Camera return description must be NEUTRAL: 'the camera returns to the original viewpoint' — NEVER state what the final state is.",
  "camera_WM": "video_WM with camera movement sentences removed. Rule: if occlusion IS camera-based (panning, rotation, zoom) — strip those camera movement sentences, keeping the scene description and physical conditions only. If occlusion is NOT camera-based (lights off, fog, blocking object, etc.) — return the FULL video_WM unchanged, because those scene events are NOT camera instructions and must be preserved. NEVER return an empty string."
}}
```

### Critical Requirements:

1. **Visual consistency**: Variants should use the SAME initial frame with minimal edits
   - **Occlusion-only variants**: `init_frame_edit_prompt` = "[NO CHANGE]" (reuse original frame)
   - **Condition variants**: `init_frame_edit_prompt` = "Edit the image to change [ONLY the key condition]"
   - Example: "Edit the image to show the cooktop display reading '32F' instead of '350F'"
   - Example: "Edit the image to change the ramp surface to rough wood texture instead of smooth metal"
   - **DO NOT** change camera angle, lighting, background, or unrelated objects

2. **Maintain task structure**: Keep the two-part structure (scene dynamics + occlusion)

3. **Preserve implied physics**: Don't explicitly state dynamics ("will melt", "will slide")

4. **Complete occlusion**: All variants must fully occlude the scene

5. **Natural fit**: Occlusion method and conditions should feel appropriate for the scene

6. **Testable differences**: Variants should produce observably different final states

7. **NO LEAKAGE OF FINAL STATE — CRITICAL**:
   Prompts must NEVER state, hint at, or describe the expected final state of the key variable.
   This rule applies to EVERY field: `video_WM`, `camera_WM`, and `variation_description`.

   **video_WM**:
   - The camera return sentence must be neutral: "The camera returns to the original viewpoint."
   - ❌ "The camera returns to reveal the ice is now a puddle of water."
   - ✅ "The camera returns to the original viewpoint."

   **camera_WM**:
   - Describe the initial physical setup and occlusion method only. Stop before any outcome.
   - ❌ "An ice cube sits on a red-hot stove burner, steaming and melting rapidly into a pool of water. STOP."
   - ✅ "An ice cube sits on a red-hot stove burner. STOP."

   **Conditions vs. outcomes**:
   - ❌ "The cooktop is turned off so the ice will not melt."
   - ✅ "The cooktop is turned off." (state the condition; let the model infer the consequence)

### Output Format:

Return ONLY valid JSON (no markdown, no commentary):

```json
{{
  "original_task_id": "...",
  "variants": [
    {{ ... variant 1 ... }},
    {{ ... variant 2 ... }},
    ...
  ]
}}
```

---

## ORIGINAL TASK

Task ID: {task_id}
Level: {level}
Category: {category}

**Original initial frame**: [See attached image]

**video_WM**:
\"\"\"{video_WM}\"\"\"

**camera_WM**:
\"\"\"{camera_WM}\"\"\"

---

Now generate {num_variants} variants exploring both occlusion methods and dynamic conditions.

Remember: You are seeing the ORIGINAL initial frame. For each variant, decide if it needs editing or can use "[NO CHANGE]".
"""


LEVEL5_MASTER_PROMPT = """
# StateWMBench Level 5: Causal Mechanism Task Variant Generation

## Overview

Level 5 tasks test whether a world model understands **functional causal relationships**:
a trigger mechanism (switch, button, pull-string, valve, key) causes a state change in a
spatially separated output (lamp, door, motor, display).

The task structure is:
- Camera starts on the OUTPUT object
- Camera pans to the TRIGGER mechanism (action is performed here while output is off-screen)
- Camera returns to the OUTPUT to reveal whether the state changed

Because trigger and output are spatially separated, the interesting variation in Level 5
is NOT in the occlusion method, but in the **causal state** of the system.

## Four Axes of Variation

### Axis A — Initial State / Toggle Direction

The same trigger action produces the OPPOSITE outcome when the initial state is reversed.

Examples:
- Lamp starts ON → pulling string turns it OFF
- Switch starts in ON position → flipping it turns the light OFF
- Valve starts open → rotating it closes the flow

Requirements:
- Edit the initial frame to show the output in the ACTIVE/ON state instead of inactive/OFF
- Keep the camera structure and trigger action identical to the original
- The final state should be the OPPOSITE of the original (e.g. light OFF instead of ON)
- `init_frame_edit_prompt` must change the output to its active/on state

---

### Axis B — Precondition Failure

A silent broken precondition prevents the causal chain from completing, so the trigger
action has NO effect on the output.

Examples:
- Power cord unplugged → flipping switch has no effect on the light
- Circuit breaker tripped → toggle produces no change
- Card is demagnetized → reader does not unlock
- Battery dead → remote control does not respond
- Safety interlock engaged → mechanism cannot activate

Requirements:
- Edit the initial frame to show ONE clear visual indicator of the broken precondition
  (e.g. unplugged cord, tripped breaker, cracked card, depleted battery indicator)
- Keep the trigger action identical — the hand still performs it
- The final state should be UNCHANGED from the initial state
- `init_frame_edit_prompt` must add the visual indicator of the failure condition

---

### Axis C — Repeated Activation (Toggle Return)

For toggle mechanisms, activating the trigger TWICE while the output is occluded returns
the output to its original state — a net no-change.

Examples:
- Pull string twice → lamp returns to original OFF state (OFF → ON → OFF)
- Toggle switch twice → light is same as at start
- Push-push mechanism activated twice → net no change

Requirements:
- `init_frame_edit_prompt` = "[NO CHANGE]" (same initial state as original)
- `video_WM` must describe the hand performing the trigger action TWICE while the output
  is occluded
- The final state should match the initial state (net no change)

---

### Axis D — Indirect / Mediated Causation

Insert a visible intermediate element between trigger and output, making the causal chain
longer or requiring an additional condition to be met before the output changes.

Examples:
- Trigger → visible relay or timer indicator → output (output only changes after delay)
- Switch controls an intermediate status light that in turn enables the output lamp
- Mechanism requires TWO independent inputs both satisfied before output changes
- A visible fuse or circuit element that must be intact for the chain to complete

Requirements:
- Edit the initial frame to add a visible intermediate element or a second required input
- The `video_WM` should reference the intermediate element naturally
- The causal relationship must still be IMPLIED, not explicitly stated

---

## Your Task

You are given an ORIGINAL Level 5 task. Generate {num_variants} variants using the axes above.

### Variant Distribution (approximate):
- **{axis_a_variants} variants**: Axis A — initial state / toggle direction
- **{axis_b_variants} variants**: Axis B — precondition failure
- **{axis_c_variants} variants**: Axis C — repeated activation
- **{axis_d_variants} variants**: Axis D — indirect / mediated causation

### For Each Variant, Provide:

```json
{{
  "variant_id": "original_task_id_02",
  "variant_suffix": "initial_state_on",
  "variation_type": "initial_state | precondition_failure | repeated_activation | indirect_causation",
  "variation_description": "Brief description of what changed",
  "changes": {{
    "occlusion_method": null,
    "dynamic_conditions": "Description of changed causal state or structure",
    "expected_outcome_change": "What the final state should be and why it differs from original"
  }},
  "init_frame_edit_prompt": "Edit the image to ... (or '[NO CHANGE]' for Axis C)",
  "video_WM": "Full modified video prompt. Camera return sentence must be NEUTRAL — see NO-LEAK rules below.",
  "camera_WM": "video_WM with camera movement sentences removed. Rule: if occlusion IS camera-based (panning, rotation, zoom) — strip those camera movement sentences, keeping the scene description and physical conditions only. If occlusion is NOT camera-based (lights off, fog, blocking object, etc.) — return the FULL video_WM unchanged, because those scene events are NOT camera instructions and must be preserved. NEVER return an empty string. See NO-LEAK rules below."
}}
```

### NO-LEAK REQUIREMENT — CRITICAL

The benchmark tests whether the world model can infer the final state from physics and causal
reasoning. If the prompt states the outcome, the task is invalid. This is the most common
error in Level 5 variant generation — you MUST follow these rules for every variant.

**In `video_WM`**:
- The camera return sentence must be completely neutral. It reveals the object, not the state.
- ❌ "revealing that the ceiling light has turned OFF and the room is dark"
- ❌ "bringing the now-illuminated ceiling light back into view"
- ✅ "bringing the ceiling light back into view"
- ❌ "while the light is out of view, a human hand flips the switch DOWN to the OFF position"
- ✅ "while the light is out of view, a human hand toggles the wall switch"
  (describe the action mechanically, not the direction that encodes the outcome)

**In `camera_WM`**:
- Describe only camera movement and the action performed. Never the result.
- ❌ "camera pans back to reveal the light is now OFF"
- ❌ "camera returns to the light, which is now illuminated"
- ✅ "camera pans from the ceiling light to the wall switch; a hand toggles the switch; camera pans back to the ceiling light. STOP."

**In `changes.expected_outcome_change`**:
- This field is for internal bookkeeping only and is NOT shown to the model.
  You may describe the expected outcome here clearly. It is exempt from the no-leak rule.

**In `variation_description`**:
- Describe what CHANGED in the setup, not what the outcome will be.
- ❌ "Light starts ON so toggling the switch will turn it OFF"
- ✅ "Light and switch both start in the active/ON state"

### Critical Requirements:

1. **Keep the camera structure unchanged**: The camera rotation (output → trigger → output)
   stays the same for all variants. Set `occlusion_method` to null in `changes`.

2. **Minimal init frame edits**: Only change what is strictly necessary for the variant axis.
   Do not alter camera angle, lighting, or unrelated objects.

3. **Implied causality only**: State conditions, never outcomes.
   - ❌ "the power cord is unplugged so the light will not turn on"
   - ✅ "the power cord is unplugged"

4. **Testable final states**: Each variant should produce a clearly different (or clearly
   same) final state compared to the original, verifiable by a judge VLM.

5. **[NO CHANGE] for Axis C only**: Axis C reuses the original frame exactly.

### Output Format:

Return ONLY valid JSON (no markdown, no commentary):

```json
{{
  "original_task_id": "...",
  "variants": [
    {{ ... variant 1 ... }},
    ...
  ]
}}
```

---

## ORIGINAL TASK

Task ID: {task_id}
Level: {level}
Category: {category}

**Original initial frame**: [See attached image]

**video_WM**:
\"\"\"{video_WM}\"\"\"

**camera_WM**:
\"\"\"{camera_WM}\"\"\"

---

Now generate {num_variants} variants for this Level 5 task using Axes A–D.
"""


def infer_hy_worldplay(video_wm: str) -> Optional[str]:
    """Infer the HY-WorldPlay camera control string from a video_WM prompt.

    Returns:
        "s-3, right-30, left-30"  — camera pans right first, then left back
        "s-3, left-30, right-30"  — camera pans left first, then right back
        "s-31, w-32"              — camera zooms out then back in
        None                      — no camera movement detected
    """
    lower = (video_wm or "").lower()

    zoom_phrases = [
        "zooms out", "zoom out", "zooming out",
    ]
    right_phrases = [
        "rotates right", "moves right", "pans right",
        "rotate right", "move right", "pan right", "moving right",
    ]
    left_phrases = [
        "rotates left", "moves left", "pans left",
        "rotate left", "move left", "pan left", "moving left",
    ]

    if any(p in lower for p in zoom_phrases):
        return "s-31, w-32"

    right_pos = min(
        (lower.find(p) for p in right_phrases if lower.find(p) != -1),
        default=-1,
    )
    left_pos = min(
        (lower.find(p) for p in left_phrases if lower.find(p) != -1),
        default=-1,
    )

    if right_pos == -1 and left_pos == -1:
        return None
    if right_pos != -1 and (left_pos == -1 or right_pos < left_pos):
        return "s-3, right-30, left-30"
    return "s-3, left-30, right-30"


def strip_camera_sentences(video_wm: str) -> str:
    """Remove camera-movement sentences from a video_WM prompt to produce camera_WM.

    Used as a fallback when Gemini returns an empty camera_WM for a camera-based
    occlusion variant. Strips only sentences that begin with explicit camera movement
    instructions ("the camera", "camera "). Non-camera occlusion (lights off, fog,
    blocking objects) is left unchanged — those are scene events, not camera directives.
    """
    if not video_wm:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", video_wm.strip())

    camera_starters = ("the camera", "camera ")

    kept = [
        s for s in sentences
        if not any(s.lower().strip().startswith(kw) for kw in camera_starters)
    ]

    return " ".join(kept).strip() if kept else video_wm.strip()


def call_gemini_for_variants(
    api_key: str,
    model: str,
    task_id: str,
    level: int,
    category: str,
    initial_frame_path: Path,
    video_wm: str,
    camera_wm: str,
    num_variants: int,
    temperature: float,
) -> Dict[str, Any]:
    """Call Gemini to generate task variants."""

    # Calculate variant distribution
    occlusion_variants = max(2, num_variants // 3)
    condition_variants = max(2, num_variants // 3)
    combined_variants = num_variants - occlusion_variants - condition_variants

    prompt = MASTER_PROMPT.format(
        task_id=task_id,
        level=level,
        category=category,
        video_WM=video_wm,
        camera_WM=camera_wm,
        num_variants=num_variants,
        occlusion_variants=occlusion_variants,
        condition_variants=condition_variants,
        combined_variants=combined_variants,
    )

    client = genai.Client(api_key=api_key)

    # Include the original initial frame image
    contents = [
        image_to_inline_data(initial_frame_path),
        {"text": prompt}
    ]

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config={"temperature": temperature},
    )

    raw = getattr(resp, "text", None) or str(resp)
    data = extract_json(raw)

    if "variants" not in data or not isinstance(data["variants"], list):
        raise ValueError("Gemini output missing 'variants' list.")

    return data


def call_gemini_for_level5_variants(
    api_key: str,
    model: str,
    task_id: str,
    level: int,
    category: str,
    initial_frame_path: Path,
    video_wm: str,
    camera_wm: str,
    num_variants: int,
    temperature: float,
) -> Dict[str, Any]:
    """Call Gemini to generate Level 5 causal variants using Axes A-D."""

    # Distribute variants across the four axes
    axis_a = max(1, round(num_variants * 0.30))  # initial state
    axis_b = max(1, round(num_variants * 0.35))  # precondition failure
    axis_c = max(1, round(num_variants * 0.20))  # repeated activation
    axis_d = max(1, num_variants - axis_a - axis_b - axis_c)  # indirect causation

    prompt = LEVEL5_MASTER_PROMPT.format(
        task_id=task_id,
        level=level,
        category=category,
        video_WM=video_wm,
        camera_WM=camera_wm,
        num_variants=num_variants,
        axis_a_variants=axis_a,
        axis_b_variants=axis_b,
        axis_c_variants=axis_c,
        axis_d_variants=axis_d,
    )

    client = genai.Client(api_key=api_key)

    contents = [
        image_to_inline_data(initial_frame_path),
        {"text": prompt},
    ]

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config={"temperature": temperature},
    )

    raw = getattr(resp, "text", None) or str(resp)
    data = extract_json(raw)

    if "variants" not in data or not isinstance(data["variants"], list):
        raise ValueError("Gemini output missing 'variants' list.")

    return data


# -----------------------------
# Fully observable baseline
# -----------------------------

def rewrite_prompt_for_full_observability(
    video_wm: str,
    api_key: str,
    model: str = "gemini-2.0-flash-exp",
    temperature: float = 0.2,
) -> str:
    """Use Gemini to rewrite video_WM prompt to remove all occlusion methods."""

    rewrite_prompt = f"""You are helping create a fully observable baseline variant of a video generation prompt.

ORIGINAL PROMPT:
{video_wm}

TASK:
Rewrite this prompt to make the scene FULLY OBSERVABLE throughout the entire video. Specifically:
1. Remove ALL camera movement (panning, rotation, zooming, etc.) - make the camera completely stationary
2. Remove ANY other occlusion methods (objects blocking view, going off-screen, etc.)
3. Keep ALL the physical dynamics and actions in the scene exactly as described
4. Keep the scene composition and framing the same
5. The result should be a continuous shot with a stationary camera where everything remains visible

GUIDELINES:
- If the prompt describes camera movement, replace it with "Camera remains stationary throughout"
- If the prompt describes objects going out of view, remove that and keep them in view
- Preserve all physical processes, state changes, and actions
- Keep the same writing style and level of detail
- Do NOT add new elements or dynamics
- Output ONLY the rewritten prompt, no explanations or commentary

NO-LEAK REQUIREMENT — CRITICAL:
The rewritten prompt must NEVER state, hint at, or describe the expected final state of the key variable.
The prompt sets up the conditions; the world model must infer what happens.
- ❌ "The ice cube melts into a puddle of water."
- ✅ "An ice cube sits on the red-hot burner. Camera remains stationary throughout."
- ❌ "The block slides down and comes to rest at the bottom."
- ✅ "A block rests at the top of the ramp. Camera remains stationary throughout."
Describe the INITIAL STATE and CONDITIONS only. Do not narrate the outcome.

REWRITTEN PROMPT:"""

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=[{"text": rewrite_prompt}],
            config={"temperature": temperature},
        )

        rewritten = getattr(resp, "text", None) or str(resp)
        return rewritten.strip()

    except Exception as e:
        print(f"[WARN] Gemini API call failed: {e}")
        print(f"[WARN] Falling back to simple camera removal")
        # Fallback: simple append
        return f"{video_wm.rstrip()} Camera remains stationary throughout."


def rewrite_level5_for_full_observability(
    video_wm: str,
    initial_frame_path: Path,
    api_key: str,
    model: str = "gemini-2.0-flash-exp",
    temperature: float = 0.2,
) -> tuple:
    """For Level 5 tasks, generate an init_frame_edit_prompt and a new video_WM.

    Level 5 tasks have spatially separated trigger and output (e.g. wall switch
    and ceiling light). A stationary-camera baseline requires adding the trigger
    mechanism into the existing frame so both elements are visible together.
    Frame dimensions must not change (world models have fixed input requirements).

    Returns: (init_frame_edit_prompt, video_WM)
    """

    prompt = f"""You are helping create a fully observable baseline for a Level 5 causal mechanism task.

ORIGINAL VIDEO PROMPT:
{video_wm}

CONTEXT:
This is a Level 5 causal task. The scene has TWO spatially separated elements:
1. OUTPUT OBJECT: the object whose state changes (e.g. a lightbulb, a lamp, a door).
2. TRIGGER MECHANISM: the control that causes the change (e.g. a wall switch, a pull-string, a button).

The original prompt uses camera movement to pan from the output to the trigger, perform the causal
action while the output is off-screen, then pan back to reveal the state change.

YOUR TASK:
Produce TWO artifacts so the scene becomes FULLY OBSERVABLE:

1. INIT_FRAME_EDIT_PROMPT
   An image editing prompt (starting with "Edit the image to ...") that adds the trigger mechanism
   into the existing frame so that BOTH the output object AND the trigger are visible together.
   - Keep the frame size and camera viewpoint EXACTLY as-is. Do NOT widen, crop, or zoom.
     World models have fixed input frame dimensions — the frame must not change.
   - Add the trigger mechanism naturally into the existing scene
     (e.g. add a wall switch on a visible wall, add a pull-string near the lamp,
      add a button on a nearby panel).
   - Keep the original lighting, style, and all existing objects unchanged.
   - The trigger should look like it belongs in the scene and be clearly identifiable.

2. VIDEO_WM
   Rewrite the video prompt for the scene where both elements are visible in the same frame:
   - Camera is COMPLETELY STATIONARY throughout
   - Both the output object AND the trigger are visible at all times
   - A hand performs the same trigger action as in the original
   - No camera movement, no occlusion, no panning
   - Preserve all causal relationships and implied physics from the original
   - End with "STOP."

NO-LEAK REQUIREMENT — CRITICAL:
The VIDEO_WM must NEVER state, hint at, or describe the expected final state of the output.
The prompt sets up the conditions and action; the world model must infer the outcome.
- ❌ "A hand flips the switch, turning the light on."
- ✅ "A hand toggles the wall switch."
- ❌ "The ceiling light turns on, illuminating the room."
- ✅ "The ceiling light and wall switch are both visible. A hand reaches in and toggles the switch. STOP."
Describe the INITIAL STATE and the TRIGGER ACTION only. Do not narrate what the output does in response.

Return ONLY valid JSON (no markdown, no commentary):
{{
  "init_frame_edit_prompt": "Edit the image to ...",
  "video_WM": "..."
}}

Carefully identify the output object and trigger mechanism from the original prompt before writing."""

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=[
                image_to_inline_data(initial_frame_path),
                {"text": prompt},
            ],
            config={"temperature": temperature},
        )

        raw = getattr(resp, "text", None) or str(resp)
        data = extract_json(raw)

        init_frame_edit = data.get("init_frame_edit_prompt", "").strip()
        video_wm_new = data.get("video_WM", "").strip()

        if not init_frame_edit or not video_wm_new:
            raise ValueError("Gemini output missing init_frame_edit_prompt or video_WM.")

        return init_frame_edit, video_wm_new

    except Exception as e:
        print(f"[WARN] Level 5 baseline rewrite failed: {e}")
        print(f"[WARN] Falling back to generic camera removal (init frame unchanged)")
        fallback_wm = rewrite_prompt_for_full_observability(
            video_wm=video_wm, api_key=api_key, model=model
        )
        return "[NO CHANGE]", fallback_wm


def create_fully_observable_baseline(
    original_task: Dict[str, Any],
    task_id: str,
    api_key: str,
    model: str = "gemini-3-pro-preview",
    initial_frame_path: Path = None,
) -> Dict[str, Any]:
    """Create fully observable baseline variant (v0) with no occlusion.

    For Level 0-4: the trigger and result are collocated, so removing camera
    movement is sufficient and the init frame is reused unchanged.

    For Level 5: the trigger and output are spatially separated. The init frame
    must be widened to include both, and the video prompt rewritten accordingly.
    """

    level = original_task.get("level", 0)
    prompts = original_task.get("prompts", {})
    video_wm = prompts.get("video_WM", "")

    if level == 5 and initial_frame_path is not None:
        init_frame_edit_prompt, baseline_video_wm = rewrite_level5_for_full_observability(
            video_wm=video_wm,
            initial_frame_path=initial_frame_path,
            api_key=api_key,
            model=model,
        )
    else:
        init_frame_edit_prompt = "[NO CHANGE]"
        baseline_video_wm = rewrite_prompt_for_full_observability(
            video_wm=video_wm,
            api_key=api_key,
            model=model,
        )

    baseline_variant = {
        "variant_id": f"{task_id}_00",
        "variant_suffix": "fully_observable",
        "variation_type": "full_observability_baseline",
        "variation_description": "Fully observable baseline - camera remains stationary, no occlusion, always visible",
        "changes": {
            "occlusion_method": "No occlusion - camera remains stationary throughout",
            "dynamic_conditions": None,
            "expected_outcome_change": "Same dynamics as original, but scene remains visible at all times"
        },
        "init_frame_edit_prompt": init_frame_edit_prompt,
        "video_WM": baseline_video_wm,
        "camera_WM": baseline_video_wm,
    }

    return baseline_variant


# -----------------------------
# Variant saving
# -----------------------------

def create_variant_yaml(
    original_task: Dict[str, Any],
    variant: Dict[str, Any],
    original_task_id: str,
) -> Dict[str, Any]:
    """Create a new task YAML from a variant specification."""

    new_task = {
        "id": variant["variant_id"],
        "level": original_task.get("level"),
        "category": original_task.get("category"),
        "tags": original_task.get("tags", []),
        "variant_of": original_task_id,
        "variation_type": variant["variation_type"],
        "variation_description": variant["variation_description"],
        "changes": variant["changes"],
    }

    init_frame_edit = variant.get("init_frame_edit_prompt", "[NO CHANGE]")

    # Infer camera pan direction directly from the variant's video_WM.
    # This is the ground truth: occlusion_method descriptions can be vague, but
    # the actual prompt always says which direction the camera moves first.
    hy_worldplay = infer_hy_worldplay(variant.get("video_WM", ""))

    # Resolve camera_WM: use Gemini-provided value when present; otherwise fall back.
    # Definition: camera_WM = video_WM minus camera-based movement instructions.
    camera_wm = (variant.get("camera_WM") or "").strip()
    if not camera_wm:
        if hy_worldplay is None:
            # No camera movement in this variant (lights, fog, blocking object, etc.).
            # Non-camera occlusion is a scene event, not a camera instruction, so
            # camera_WM = video_WM exactly.
            camera_wm = variant.get("video_WM", "")
        else:
            # Camera-based occlusion — strip camera movement sentences.
            # Borrow original task's camera_WM first (same camera structure, same scene).
            camera_wm = ((original_task.get("prompts") or {}).get("camera_WM") or "").strip()
            if not camera_wm:
                camera_wm = strip_camera_sentences(variant.get("video_WM", ""))

    new_task["prompts"] = {
        "init_frame_edit_prompt": init_frame_edit,
        "video_WM": variant["video_WM"],
        "camera_WM": camera_wm or None,
    }

    # Set camera_control from the inferred pan direction.
    # hy_worldplay is None when there is no camera movement (baseline, blocking, etc.).
    if "camera_control" in original_task:
        new_task["camera_control"] = {
            key: (hy_worldplay if key == "HY-WorldPlay" else None)
            for key in original_task["camera_control"]
        }
    else:
        new_task["camera_control"] = {"HY-WorldPlay": hy_worldplay, "Genie": None}

    # Don't copy evaluation - will be regenerated

    return new_task


def save_variants(
    level_dir: Path,
    original_task_id: str,
    original_task_dir: Path,
    original_task: Dict[str, Any],
    variants: List[Dict[str, Any]],
    dry_run: bool = False,
) -> List[Path]:
    """Save variant YAMLs to new task directories."""

    created_paths = []

    # Locate original init frame
    original_init_frame = original_task_dir / f"{original_task_id}_init_frame.png"
    if not original_init_frame.exists():
        print(f"Warning: Original init frame not found: {original_init_frame}")
        original_init_frame = None

    for variant in variants:
        variant_id = variant["variant_id"]
        init_frame_edit = variant.get("init_frame_edit_prompt", "")

        # Create variant directory
        variant_dir = level_dir / variant_id

        if dry_run:
            print(f"[DRY RUN] Would create: {variant_dir}")
            print(f"  - init_frame: {init_frame_edit[:50]}...")
            continue

        variant_dir.mkdir(exist_ok=True)

        # Create variant YAML
        variant_task = create_variant_yaml(original_task, variant, original_task_id)
        variant_yaml_path = variant_dir / f"{variant_id}.yaml"

        dump_yaml(variant_yaml_path, variant_task)
        print(f"  Created variant YAML: {variant_yaml_path.name}")

        # Handle initial frame
        if original_init_frame and original_init_frame.exists():
            variant_init_frame = variant_dir / f"{variant_id}_init_frame.png"

            if init_frame_edit.strip().upper() == "[NO CHANGE]":
                # No editing needed — copy with the variant's own filename.
                shutil.copy2(original_init_frame, variant_init_frame)
                print(f"    -> Copied original init frame (no changes needed)")
            else:
                # Copy with the ORIGINAL task's filename as the placeholder.
                # generate_init_frame.py identifies frames that need editing by
                # checking whether the file's base name differs from the task id.
                # Using the original filename ensures Situation 2 is triggered.
                placeholder_path = variant_dir / original_init_frame.name
                shutil.copy2(original_init_frame, placeholder_path)
                print(f"    -> Copied original as placeholder: {original_init_frame.name}")
                print(f"    -> Run generate_init_frame.py to apply: {init_frame_edit[:60]}...")

        created_paths.append(variant_yaml_path)

    return created_paths


# -----------------------------
# Main
# -----------------------------

def get_task_level(task_dir: Path) -> Optional[int]:
    """Peek at a task YAML to read its level field. Returns None if unreadable."""
    try:
        folder_name = task_dir.name
        task_yaml_path = task_dir / f"{folder_name}.yaml"
        if not task_yaml_path.exists():
            return None
        task = load_yaml(task_yaml_path)
        return task.get("level")
    except Exception:
        return None


def process_task_default(
    task_dir: Path,
    api_key: str,
    model: str,
    num_variants: int,
    temperature: float,
    dry_run: bool = False,
) -> None:
    """Process a Level 0-4 task to generate variants.

    Always creates a fully observable baseline (v0) first, then generates
    additional variants along the occlusion-method and dynamic-condition axes.
    """

    _, task_yaml_path, folder_name, initial_frame_path = resolve_task_paths(task_dir)

    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")

    if not initial_frame_path or not initial_frame_path.exists():
        print(f"  [WARN] Skipping {folder_name}: missing initial frame")
        return

    task = load_yaml(task_yaml_path)

    # Extract fields
    task_id = task.get("id", folder_name)
    level = task.get("level", 0)
    category = task.get("category", "unknown")

    prompts = task.get("prompts", {})
    video_wm = prompts.get("video_WM", "")
    camera_wm = prompts.get("camera_WM", "")

    if not video_wm:
        print(f"  [WARN] Skipping {task_id}: missing video_WM prompt")
        return

    # Step 1: Create fully observable baseline variant (v0)
    print(f" Creating fully observable baseline variant (v0)...")
    baseline_variant = create_fully_observable_baseline(
        original_task=task,
        task_id=task_id,
        api_key=api_key,
        model=model,
        initial_frame_path=initial_frame_path,
    )

    all_variants = [baseline_variant]
    print(f"  [OK] Created baseline variant: {baseline_variant['variant_id']}")

    # Step 2: Call Gemini to generate additional variants
    print(f" Calling {model} to generate {num_variants} additional variants...")

    try:
        result = call_gemini_for_variants(
            api_key=api_key,
            model=model,
            task_id=task_id,
            level=level,
            category=category,
            initial_frame_path=initial_frame_path,
            video_wm=video_wm,
            camera_wm=camera_wm,
            num_variants=num_variants,
            temperature=temperature,
        )

        variants = result["variants"]
        all_variants.extend(variants)
        print(f"  [OK] Generated {len(variants)} additional variants")
        print(f"  [OK] Total variants (including baseline): {len(all_variants)}")

        # Save all variants (baseline + generated)
        level_dir = task_dir.parent
        created = save_variants(
            level_dir, task_id, task_dir, task, all_variants, dry_run=dry_run
        )

        if not dry_run:
            print(f"  [OK] Saved {len(created)} variant YAMLs")

    except Exception as e:
        print(f"  [ERROR] Error generating variants: {e}")
        raise


def process_task_level5(
    task_dir: Path,
    api_key: str,
    model: str,
    num_variants: int,
    temperature: float,
    dry_run: bool = False,
) -> None:
    """Process a Level 5 causal mechanism task to generate variants.

    Always creates a fully observable baseline (v0) first, with the trigger
    mechanism added into the existing init frame so both trigger and output are
    visible together (frame dimensions unchanged). Additional variants
    explore the four causal axes:
      A — initial state / toggle direction
      B — precondition failure
      C — repeated activation (toggle return)
      D — indirect / mediated causation
    """

    _, task_yaml_path, folder_name, initial_frame_path = resolve_task_paths(task_dir)

    print(f"\n{'='*60}")
    print(f"Processing (Level 5): {folder_name}")
    print(f"{'='*60}")

    if not initial_frame_path or not initial_frame_path.exists():
        print(f"  [WARN] Skipping {folder_name}: missing initial frame")
        return

    task = load_yaml(task_yaml_path)

    task_id = task.get("id", folder_name)
    level = task.get("level", 5)
    category = task.get("category", "unknown")

    prompts = task.get("prompts", {})
    video_wm = prompts.get("video_WM", "")
    camera_wm = prompts.get("camera_WM", "")

    if not video_wm:
        print(f"  [WARN] Skipping {task_id}: missing video_WM prompt")
        return

    # Step 1: Fully observable baseline — widened init frame + stationary camera
    print(f"  Creating fully observable baseline (v0) [Level 5: adding trigger into existing frame]...")
    baseline_variant = create_fully_observable_baseline(
        original_task=task,
        task_id=task_id,
        api_key=api_key,
        model=model,
        initial_frame_path=initial_frame_path,
    )

    all_variants = [baseline_variant]
    print(f"  [OK] Created baseline variant: {baseline_variant['variant_id']}")

    # Step 2: Generate causal variants via Axes A–D
    print(f"  Calling {model} to generate {num_variants} Level 5 causal variants (Axes A–D)...")

    try:
        result = call_gemini_for_level5_variants(
            api_key=api_key,
            model=model,
            task_id=task_id,
            level=level,
            category=category,
            initial_frame_path=initial_frame_path,
            video_wm=video_wm,
            camera_wm=camera_wm,
            num_variants=num_variants,
            temperature=temperature,
        )

        variants = result["variants"]
        all_variants.extend(variants)
        print(f"  [OK] Generated {len(variants)} causal variants")
        print(f"  [OK] Total variants (including baseline): {len(all_variants)}")

        level_dir = task_dir.parent
        created = save_variants(
            level_dir, task_id, task_dir, task, all_variants, dry_run=dry_run
        )

        if not dry_run:
            print(f"  [OK] Saved {len(created)} variant YAMLs")

    except Exception as e:
        print(f"  [ERROR] Error generating Level 5 variants: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate task variants along occlusion and dynamic condition axes"
    )
    parser.add_argument(
        "--level_dir",
        type=str,
        help="Level directory (e.g., benchmark/tasks/level1_scalar_state)",
    )
    parser.add_argument(
        "--task_path",
        type=str,
        help="Single task directory or YAML to process (alternative to --level_dir)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-pro-preview",
        # default="gemini-2.5-pro",
        type=str,
        help="Gemini model to use",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="Temperature for generation (higher = more creative)",
    )
    parser.add_argument(
        "--num_variants",
        default=10,
        type=int,
        help="Number of variants to generate per task",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually creating files",
    )

    args = parser.parse_args()

    if not args.level_dir and not args.task_path:
        parser.error("Must specify either --level_dir or --task_path")

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

    if args.task_path:
        # Process single task
        task_path = Path(args.task_path).resolve()
        task_dir = task_path.parent if task_path.is_file() else task_path
        level = get_task_level(task_dir)
        runner = process_task_level5 if level == 5 else process_task_default
        runner(
            task_dir,
            api_key,
            args.model,
            args.num_variants,
            args.temperature,
            args.dry_run,
        )
    else:
        # Process all tasks in level directory
        level_dir = Path(args.level_dir).resolve()

        if not level_dir.exists() or not level_dir.is_dir():
            raise FileNotFoundError(f"Level directory not found: {level_dir}")

        task_dirs = [d for d in level_dir.iterdir() if d.is_dir()]
        print(f"Found {len(task_dirs)} tasks in {level_dir.name}")

        for task_dir in sorted(task_dirs):
            try:
                level = get_task_level(task_dir)
                runner = process_task_level5 if level == 5 else process_task_default
                runner(
                    task_dir,
                    api_key,
                    args.model,
                    args.num_variants,
                    args.temperature,
                    args.dry_run,
                )
            except Exception as e:
                print(f"  x Failed to process {task_dir.name}: {e}")
                continue

        print(f"\n{'='*60}")
        print(" Variant generation complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
