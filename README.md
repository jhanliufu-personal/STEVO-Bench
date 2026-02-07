# World-Model Benchmark: Stateful Dynamics Under Partial Observability

This repository contains a benchmark for evaluating **stateful world models**—models that must maintain, update, and reason over **latent world state** when observations are partial, delayed, or occluded.

The benchmark is designed to go beyond short-horizon visual prediction and probe whether a model internally represents **objects, relations, transformations, and causal mechanisms**, rather than relying on frame-to-frame pattern completion.

---

## Motivation

Recent video and multimodal generative models can produce visually convincing rollouts, yet often fail in subtle but fundamental ways:

- objects reappear where they started instead of where they should be,
- quantities reset after occlusion,
- containers forget what they contain,
- structures “melt” instead of breaking,
- causal interventions have no effect unless explicitly visible.

These failures suggest that many models do not maintain a coherent **latent world state** over time.

This benchmark aims to systematically test *what kinds of state* a model can represent and update, using controlled tasks with **partial observability**, **occlusion**, and **off-screen interventions**.

---

## Conceptual Framework

We model each task as a **Partially Observable Decision Process over World Dynamics (PODWD)**:

\[
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{O}, T, Z \rangle
\]

- \(\mathcal{S}\): latent world state (not fully observable)
- \(\mathcal{A}\): actions (camera motion, interventions)
- \(\mathcal{O}\): observations (video frames)
- \(T\): world transition dynamics
- \(Z\): observation function (what is visible)

The benchmark evaluates whether a model maintains a **belief state** over \(\mathcal{S}\) that remains consistent across time and occlusion.

Crucially, tasks are organized by the **minimal latent state structure** required for correct inference. Each level strictly adds representational demands.

---

## Taxonomy of World Dynamics

The benchmark taxonomy is progressive and non-overlapping.  
Each level answers a deeper question about *what the model must internally track*.

Below, each category includes:
- a **formal description** (what state is required),
- an **intuitive description** (what this means in the world),
- placeholders for **example tasks and outputs**.

---

### **Level 0 — Object Existence & Identity**
**Formal:** Persistent entity identity across time  
**Latent state:** object IDs and basic attributes

**Intuition:**  
Do objects continue to exist when you can’t see them?

This is the most basic requirement of a world model. Without object permanence, nothing else is meaningful.

**Typical challenges:**
- objects disappearing after occlusion
- objects reappearing at their original position

**Example tasks:**  
- Shell game (object hidden under moving cup)  
- Occlusion and reveal  

**Example outputs:**  
*(to be filled)*

---

### **Level 1 — Continuous Attribute Evolution**
**Formal:** Per-object continuous latent variables integrated over time  
**Latent state:** scalars such as volume, height, temperature, charge

**Intuition:**  
Can the model track *how much* of something there is when it’s not visible?

The object stays the same, but some internal quantity changes while off-screen.

**Typical challenges:**
- quantities freezing during occlusion
- resetting to initial values on reappearance

**Example tasks:**  
- Tank filling with water  
- Sand accumulating in a container  

**Example outputs:**  
*(to be filled)*

---

### **Level 2 — Spatial Kinematics & Constraints**
**Formal:** Geometric state with motion and constraints  
**Latent state:** position, velocity, orientation, constraint variables

**Intuition:**  
Can the model predict *where things go* when they move out of sight?

This level requires spatial reasoning, not just scalar integration.

**Typical challenges:**
- incorrect re-entry position
- violation of constraints (e.g., objects leaving rails)

**Example tasks:**  
- Sliding objects on ramps  
- Pendulum motion under occlusion  

**Example outputs:**  
*(to be filled)*

---

### **Level 3 — Interaction & Relational Transfer**
**Formal:** Graph-structured state with coupled object dynamics  
**Latent state:** relations such as containment, attachment, connectivity

**Intuition:**  
Can the model track *who has what*, *what is connected to what*, or *what affects what*?

Here, objects cannot be modeled independently—the state lives in relations.

**Typical challenges:**
- forgetting contents of containers
- breaking conservation across entities
- failing to propagate effects through contact

**Example tasks:**  
- Objects loaded into a box  
- Track switching routes an object  
- Domino chains  

**Example outputs:**  
*(to be filled)*

---

### **Level 4 — Structural & Ontological Transformations**
**Formal:** Hybrid state with discrete mode changes and state rewrites  
**Latent state:** object topology, integrity, count, phase

**Intuition:**  
Can the model represent **irreversible change**?

This is not “less of the same object,” but *a different world* after an event.

**Typical challenges:**
- shattering rendered as melting
- broken objects reappearing intact
- no event segmentation

**Example tasks:**  
- Glass shattering  
- Block tower collapse  
- Object cut into pieces  

**Example outputs:**  
*(to be filled)*

---

### **Level 5 — Mechanism-Level & Rule-Based Causality**
**Formal:** Latent variables that govern transition rules themselves  
**Latent state:** hidden mechanism or configuration variables

**Intuition:**  
Can the model reason about **how the world works**, not just what it looks like?

Two scenes may look identical, but behave differently depending on an unseen switch, latch, or routing.

**Typical challenges:**
- ignoring off-screen interventions
- failing to apply deterministic rules
- treating causal changes as visual noise

**Example tasks:**  
- Light controlled by hidden switch  
- Door opens only if latch unlocked  
- Track switch determines object path  

**Example outputs:**  
*(to be filled)*

---

## Benchmark Philosophy

- **Occlusion is essential**: important state changes often happen while unseen.
- **Interventions are explicit**: the challenge is inferring consequences, not guessing actions.
- **Initial frames are canonical inputs**: generated videos are model outputs.
- **Outputs are not stored in the repo**: evaluation is reproducible without fixing generations.

The benchmark is intended to be **diagnostic**, not just comparative:
failures are often more informative than scores.

---

## Repository Structure

See `benchmark/`, `tasks/`, and `eval/` for detailed organization of task specifications, prompts, camera scripts, and evaluation code.

Generated videos and model outputs are expected to be stored **locally or in external artifact storage**, not committed to this repository.

---

## Citation

If you use this benchmark, please cite:
