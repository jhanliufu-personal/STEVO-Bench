#!/usr/bin/env python3
"""
Generate task variants along two axes:
1. Occlusion method (camera rotation → object blocking, zoom out, etc.)
2. Dynamic conditions (key variables that affect state evolution)

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
import base64
import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from google import genai

from utils import load_yaml, dump_yaml, resolve_task_paths

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
- Example: "A cooktop displaying 350°F" → implies heat, but doesn't say "the ice will melt"
- Example: "A smooth, frictionless ramp" → implies sliding, but doesn't say "the block will accelerate"

**Part B - Camera Movement (Creates Occlusion)**:
- Camera moves to occlude the scene
- Critical state evolution happens OFF-SCREEN
- Camera returns to reveal final state
- Standard template: "The camera rotates right until [object] completely exits the view, then rotates toward the left smoothly at the same rate until [location] is back in the center of view. No zoom, no cut, no scene change. STOP."

### 3. Camera WM Prompt (`camera_WM`)
Simplified version focusing on the physical event without camera instructions.

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
- Surface friction (frictionless → high friction)
- Initial velocity (slow → fast)
- Mass (light → heavy)
- Angle of incline (steep → shallow)
- Air resistance (vacuum → normal air)

**For thermal tasks** (melting, heating, cooling):
- Temperature differential (hot surface → lukewarm)
- Initial object state (frozen solid → partially thawed)
- Thermal conductivity (metal surface → wood surface)
- Ambient conditions (hot room → cold room)

**For phase change tasks** (inflating, deflating, dissolving):
- Rate of change (fast pump → slow leak)
- Initial state (empty → partially filled)
- Pressure differential (high → low)
- Concentration (pure → diluted)

**For relational tasks** (pulleys, collisions, reflections):
- Mass ratios (equal masses → unequal)
- Coupling strength (tight → loose)
- Symmetry (symmetric → asymmetric)
- Initial energy (high → low)

**For causal tasks** (switches, mechanisms):
- Activation threshold (binary → gradual)
- Preconditions (already met → not met)
- State history (first pull → second pull)
- External enabling conditions (power on → power off)

**Requirements**:
- Variants should test the MODEL'S understanding of how conditions affect dynamics
- Some variants should have NO state change (e.g., ice on cold surface → no melting)
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
  "video_WM": "Modified full video prompt with new occlusion and/or conditions",
  "camera_WM": "Modified simplified prompt"
}}
```

### Critical Requirements:

1. **Visual consistency**: Variants should use the SAME initial frame with minimal edits
   - **Occlusion-only variants**: `init_frame_edit_prompt` = "[NO CHANGE]" (reuse original frame)
   - **Condition variants**: `init_frame_edit_prompt` = "Edit the image to change [ONLY the key condition]"
   - Example: "Edit the image to show the cooktop display reading '32°F' instead of '350°F'"
   - Example: "Edit the image to change the ramp surface to rough wood texture instead of smooth metal"
   - **DO NOT** change camera angle, lighting, background, or unrelated objects

2. **Maintain task structure**: Keep the two-part structure (scene dynamics + occlusion)

3. **Preserve implied physics**: Don't explicitly state dynamics ("will melt", "will slide")

4. **Complete occlusion**: All variants must fully occlude the scene

5. **Natural fit**: Occlusion method and conditions should feel appropriate for the scene

6. **Testable differences**: Variants should produce observably different final states

7. **No explicit outcomes**: Don't say "the ice will not melt" - instead say "the cooktop is turned off"

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

# -----------------------------
# Image handling
# -----------------------------

def image_to_inline_data(image_path: Path, max_side: int = 1024) -> Dict[str, Any]:
    """Convert image to inline data for Gemini API."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"inline_data": {"mime_type": "image/png", "data": data}}


# -----------------------------
# Gemini API
# -----------------------------

def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from Gemini response (handles markdown code blocks)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Could not find JSON object in Gemini response.")
    return json.loads(m.group(0))


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

    # Store the init frame edit prompt
    init_frame_edit = variant.get("init_frame_edit_prompt", "[NO CHANGE]")

    new_task["prompts"] = {
        "init_frame_edit_prompt": init_frame_edit,
        "video_WM": variant["video_WM"],
        "camera_WM": variant["camera_WM"],
    }

    # Copy camera control if present
    if "camera_control" in original_task:
        new_task["camera_control"] = original_task["camera_control"]

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
                # Copy original frame unchanged
                shutil.copy2(original_init_frame, variant_init_frame)
                print(f"    → Copied original init frame (no changes)")
            else:
                # Mark for editing (will need separate image editing step)
                print(f"    → Init frame needs editing: {init_frame_edit[:60]}...")
                # Store edit prompt in YAML for later image generation
                # Actual image editing would require a separate pass with an image editing model
                # For now, copy original and mark it for manual editing
                shutil.copy2(original_init_frame, variant_init_frame)
                print(f"    → Copied original as placeholder (run image editing separately)")

        created_paths.append(variant_yaml_path)

    return created_paths


# -----------------------------
# Main
# -----------------------------

def process_task(
    task_dir: Path,
    api_key: str,
    model: str,
    num_variants: int,
    temperature: float,
    dry_run: bool = False,
) -> None:
    """Process a single task to generate variants."""

    _, task_yaml_path, folder_name, initial_frame_path = resolve_task_paths(task_dir)

    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")

    if not initial_frame_path or not initial_frame_path.exists():
        print(f"  ⚠ Skipping {folder_name}: missing initial frame")
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
        print(f"  ⚠ Skipping {task_id}: missing video_WM prompt")
        return

    # Call Gemini to generate variants
    print(f"  🤖 Calling {model} to generate {num_variants} variants...")

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
        print(f"  ✓ Generated {len(variants)} variants")

        # Save variants
        level_dir = task_dir.parent
        created = save_variants(
            level_dir, task_id, task_dir, task, variants, dry_run=dry_run
        )

        if not dry_run:
            print(f"  ✓ Saved {len(created)} variant YAMLs")

    except Exception as e:
        print(f"  ✗ Error generating variants: {e}")
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
        "--task_dir",
        type=str,
        help="Single task directory to process (alternative to --level_dir)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-pro-preview",
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
        default=8,
        type=int,
        help="Number of variants to generate per task",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually creating files",
    )

    args = parser.parse_args()

    if not args.level_dir and not args.task_dir:
        parser.error("Must specify either --level_dir or --task_dir")

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

    if args.task_dir:
        # Process single task
        task_dir = Path(args.task_dir).resolve()
        process_task(
            task_dir,
            api_key,
            args.model,
            args.num_variants,
            args.temperature,
            args.dry_run,
        )
    else:
        # Process all tasks in level
        level_dir = Path(args.level_dir).resolve()

        if not level_dir.exists() or not level_dir.is_dir():
            raise FileNotFoundError(f"Level directory not found: {level_dir}")

        # Find all task subdirectories
        task_dirs = [d for d in level_dir.iterdir() if d.is_dir()]

        print(f"Found {len(task_dirs)} tasks in {level_dir.name}")

        for task_dir in sorted(task_dirs):
            try:
                process_task(
                    task_dir,
                    api_key,
                    args.model,
                    args.num_variants,
                    args.temperature,
                    args.dry_run,
                )
            except Exception as e:
                print(f"  ✗ Failed to process {task_dir.name}: {e}")
                continue

        print(f"\n{'='*60}")
        print("✓ Variant generation complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
