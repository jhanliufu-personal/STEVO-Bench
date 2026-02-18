# StateWMBench Taxonomy

## Overview

StateWMBench is a benchmark for evaluating video-based world models' ability to accurately maintain and evolve state representations under conditions of **partial observability**. The benchmark tests whether models can reason about hidden or occluded state by understanding physical dynamics, causal mechanisms, and temporal evolution.

## Six-Level Hierarchy

The benchmark is organized into six levels of increasing cognitive complexity:

### Level 0: Object Permanence (9 tasks)
**Category**: `object_permanence`

Tests the most fundamental capability: tracking objects when they become occluded or hidden from view.

**Key Challenge**: Model must maintain object identity and location even when the object is temporarily invisible.

**Examples**:
- `ball_and_cup_01`: Ball hidden under cup, cup moved, ball revealed
- `backpack_zip_internal_01`: Item placed in backpack and zipped
- `box_gift_reveal_01`: Object placed in box and lid closed
- `envelope_letter_sliding_01`: Letter inserted into envelope
- `safe_gold_internal_01`: Gold placed in safe and door closed

**Physics**: Simple spatial tracking with no state changes.

---

### Level 1: Scalar State (19 tasks)
**Category**: `scalar_state`

Tests tracking of continuous physical properties that change over time (volume, temperature, mass, phase, etc.).

**Key Challenge**: Model must infer state evolution based on implied physical conditions (e.g., heat, pressure, time).

**Examples**:
- `ice_on_burner_01`: Ice cube melts on hot surface
- `balloon_inflating_01`: Balloon expands as air is added
- `clay_drying_01`: Wet clay hardens over time
- `fuse_burn_01`: Fuse shortens as it burns
- `tire_inflate_01`: Tire expands with added air

**Physics**: Thermodynamics, phase changes, continuous deformation, accumulation/depletion.

---

### Level 2: Spatial State (23 tasks)
**Category**: `spatial_kinematics`

Tests tracking of position, velocity, momentum, and other kinematic properties of moving objects.

**Key Challenge**: Model must simulate motion based on initial conditions and physical laws (inertia, gravity, friction).

**Examples**:
- `air_puck_collision_01`: Frictionless puck slides across air hockey table
- `pendulum_frictionless_01`: Pendulum oscillates without damping
- `projectile_horizontal_launch_01`: Object launched horizontally
- `spring_oscillation_01`: Mass oscillates on spring
- `bowling_inertia_01`: Heavy ball rolls with constant velocity

**Physics**: Newtonian mechanics, conservation of momentum, projectile motion, harmonic oscillation.

---

### Level 3: Relational State (30 tasks)
**Category**: `relational_state`

Tests understanding of dependencies and interactions between multiple objects or state variables.

**Key Challenge**: Model must track how changes in one object's state affect another's through physical coupling.

**Examples**:
- `newtons_cradle_01`: Momentum transfer through collision
- `pulley_counterweight_01`: Two masses coupled by rope over pulley
- `mirror_reflection_01`: Object position determines reflection position
- `seesaw_balance_01`: Balance depends on relative masses and distances
- `hydraulic_syringe_01`: Pressure transferred through fluid

**Physics**: Action-reaction pairs, momentum/energy transfer, optical geometry, coupled systems.

---

### Level 4: Transformational State (12 tasks)
**Category**: `structural_transformation`

Tests understanding of **irreversible physical transformations** that fundamentally alter object structure or composition.

**Key Challenge**: Model must recognize when objects undergo permanent changes (breaking, tearing, mixing, melting).

**Examples**:
- `ceramic_plate_impact_01`: Plate shatters on impact
- `document_shredding_01`: Paper is cut into strips
- `egg_batter_merging_01`: Eggs mixed irreversibly into batter
- `rope_snap_01`: Frayed rope breaks under tension
- `sandcastle_wave_impact_01`: Sand structure collapses

**Physics**: Fracture mechanics, plastic deformation, mixing entropy, structural failure.

---

### Level 5: Causal State (11 tasks)
**Category**: `causal_mechanism`

Tests understanding of **functional causal relationships** and control mechanisms (switches, toggles, valves).

**Key Challenge**: Model must reason about abstract causal links (e.g., pulling string → light turns on) that depend on hidden internal mechanisms.

**Examples**:
- `string_pull_lamp_01`: Pull-string activates toggle switch
- `rotary_valve_supply_01`: Valve rotation controls flow
- `thermostat_threshold_dial_01`: Dial setting triggers heating element
- `keycard_reader_insertion_01`: Card insertion unlocks door
- `safety_interlock_motor_01`: Safety mechanism prevents motor start

**Physics**: Discrete state transitions, conditional logic, feedback control, electromechanical coupling.

---

## Task Structure

Each task is defined by a **YAML file** with the following structure:

### Metadata
```yaml
id: <task_id>
level: <0-5>
category: <object_permanence|scalar_state|spatial_kinematics|relational_state|structural_transformation|causal_mechanism>
tags: []  # Optional categorization
```

### Prompts

The core of each task consists of prompts for world models:

#### 1. `image_gen_prompt`
Describes the **initial scene** for image generation.

**Example**:
```yaml
image_gen_prompt: "A bright red ball sits on a wooden table. An opaque blue plastic
  cup is held by a hand directly above the ball."
```

#### 2. `video_WM` (Primary World Model Prompt)
**Two-part structure**:

**Part A - Scene Description & Dynamics**:
- Describes what happens physically
- Implies (but doesn't explicitly state) the underlying physics
- Specifies conditions that trigger state evolution (e.g., "cooktop displaying 350°F")

**Part B - Camera Movement (Creates Occlusion)**:
- Camera rotates/pans away from the scene
- Objects exit the frame completely
- Camera returns to original viewpoint
- Standard template: "The camera rotates right until [object] completely exits the view, then rotates toward the left smoothly at the same rate until [scene] is back in the center of view. No zoom, no cut, no scene change. STOP."

**Example**:
```yaml
video_WM: "The hand lowers the blue cup until it completely covers the red ball,
  then slides the cup 12 inches to the right. After the movement, the hand lifts
  the cup straight up 6 inches. The camera rotates right until the cup completely
  exits the view, then rotates toward the left smoothly at the same rate until the
  table surface where the cup was moved is back in the center of view. No zoom, no
  cut, no scene change. STOP."
```

#### 3. `camera_WM` (Simplified Version)
A shorter prompt focusing on the physical event without camera instructions.

**Example**:
```yaml
camera_WM: "The blue cup covers the red ball, moves to a new location, and is then
  lifted to reveal the contents. STOP."
```

### Camera Control
Technical parameters for specific world model systems:

```yaml
camera_control:
  HY-WorldPlay: "s-3, right-30, left-30"  # s=stationary, right/left=degrees
  Genie: null
```

### Evaluation

#### Ground Truth Description
```yaml
evaluation:
  init2final_edit_prompt: <Description of how final frame should differ from initial>
  rationale: <Explanation of why this state evolution should occur>
```

#### Judge Questions
Binary yes/no questions to evaluate model output:

```yaml
  questions:
    - id: <question_id>
      question: <Binary question about final state>
      answer_type: yes_no
      notes_for_judge: <Guidance for automated judge>
```

#### Metadata
```yaml
  generation:
    model_provider: google_gemini
    llm: gemini-3-pro-preview
    temperature: 0.2
    created_utc: '2026-02-11T19:40:47Z'
  inputs_digest:
    initial_frame_sha256: <hash>
    video_prompt_sha256: <hash>
    task_yaml_sha256_before: <hash>
  init_frame_description: <Detailed description of the actual initial frame>
```

---

## Key Design Principles

### 1. Implied vs. Explicit Physics
Prompts **imply** physical conditions without explicitly stating dynamics:
- ✅ "A cooktop displaying '350°F'" (implies heat source)
- ❌ "The ice cube will melt due to heat"

### 2. Partial Observability via Camera Movement
All tasks use **camera panning** to create occlusion:
- Camera moves away from scene
- Critical state evolution occurs off-screen
- Camera returns to reveal final state
- Tests whether model maintains state during occlusion

### 3. No Cuts or Edits
Prompts explicitly specify:
- "No zoom, no cut, no scene change"
- Ensures continuous temporal reasoning
- Prevents models from treating occlusion as a scene boundary

### 4. Binary Evaluation
Judge questions use yes/no format for objective evaluation:
- Reduces ambiguity in automated scoring
- Focuses on presence/absence of key state changes
- Includes both primary (state evolution) and secondary (consistency) checks

---

## Statistics

| Level | Category | Task Count |
|-------|----------|------------|
| 0 | Object Permanence | 9 |
| 1 | Scalar State | 19 |
| 2 | Spatial State | 23 |
| 3 | Relational State | 30 |
| 4 | Transformational State | 12 |
| 5 | Causal State | 11 |
| **Total** | | **104** |

---

## File Organization

```
benchmark/tasks/
├── level0_object_permanence/
│   ├── ball_and_cup_01/
│   │   ├── ball_and_cup_01.yaml
│   │   ├── init_frame.png
│   │   ├── final_frame_gt.png
│   │   └── ...
│   └── ...
├── level1_scalar_state/
├── level2_spatial_state/
├── level3_relational_state/
├── level4_transformational_state/
├── level5_causal_state/
└── taxonomy.md (this file)
```

Each task directory contains:
- YAML specification file
- Initial frame image
- Ground truth final frame image
- Generated videos and evaluation results
