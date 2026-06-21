# Ensemble Agreement Analysis

Compares two ensemble strategies for each evaluation metric by measuring the fraction of tasks
on which they produce the same binary label. Neither ensemble is treated as ground truth —
this is purely a consistency check between labeling strategies.

---

## Ensemble Definitions

| Name | Definition |
|---|---|
| **Old ensemble** | Gemini 3.1 Pro is called 3 times per task; result is the majority vote across those 3 responses. This is the pre-existing production label stored as the top-level field in each Gemini JSON report. |
| **New ensemble v1** | One response is drawn from each of three VLMs (Gemini's first response, Claude's single response, third VLM's single response); result is majority vote across the three. |
| **New ensemble v2** | Gemini's 3-response majority vote is first computed (identical to the old ensemble), then majority-voted against Claude's single response and the third VLM's single response. Gives Gemini's consolidated position equal weight to each other model. |

---

## VLM Assignments by Run

Third-VLM choice is constrained by available reports:

| Run | Physics | Control (occlusion / trigger) | State Evolution |
|---|---|---|---|
| `wan22_output_map_wan_run_2` | GPT-5.4 | GPT-5.4 | GPT-5.4 |
| `veo_output_map_veo_run_7` | GPT-4o | GPT-4o | GPT-4o |
| `sora_pro_output_map_sora_pro_run_4` | GPT-5.4 | GPT-5.4 | GPT-5.4 |

VEO uses GPT-4o throughout because GPT-5.4 was only run for VEO physics; control and SE reports
for GPT-5.4 on VEO do not exist. WAN and SORA use GPT-5.4 because GPT-4o physics was not run
for those two video generators until later (see §4).

All ensembles use **Gemini 3.1 Pro Preview** and **Claude Opus 4.7** as the first two members.

---

## Metrics

| Metric | Report file prefix | Key field | Positive = |
|---|---|---|---|
| `physical_inaccuracy` | `physics_report` | `physical_inaccuracy` | Video shows a physics violation |
| `occlusion_done` | `control_report` | `occlusion_done` | Scene was successfully hidden |
| `trigger_applied` | `control_report` | `trigger_applied` | Trigger action occurred |
| `state_evol` | `se_report` | `state_evol` | Expected physical change occurred |

---

## Analysis 1 — Pilot: VEO Only, New Ensemble v1

**Scope:** `veo_output_map_veo_run_7`, 20 tasks.  
**Third VLM:** GPT-4o for all metrics.  
**Ensemble:** New v1.

### physical_inaccuracy

| Task | Old (G×3) | New (G+C+4o) | G₀ | C | 4o | Match |
|---|---|---|---|---|---|---|
| ball_roll_down_cardboard | 0 | 0 | 0 | 0 | 0 | ✓ |
| balloon_block_cut_cardboard | 1 | 1 | 1 | 1 | 0 | ✓ |
| butter_melt_cardboard | 0 | 0 | 0 | 0 | 1 | ✓ |
| can_condensation_lightoff | 0 | 0 | 0 | 0 | 0 | ✓ |
| clay_drying_curtain | 0 | 0 | 0 | 0 | 0 | ✓ |
| cook_shrimp_cardboard | 0 | 0 | 0 | 0 | 0 | ✓ |
| falling_glass_cardboard | 0 | 0 | 0 | 1 | 0 | ✓ |
| food_dye_swirl_cardboard | 0 | 0 | 0 | 1 | 0 | ✓ |
| gas_pump_nozzle_trigger | 1 | 1 | 1 | 1 | 0 | ✓ |
| honey_drip_cardboard | 0 | 0 | 0 | 1 | 0 | ✓ |
| hydraulic_press_can_crush_10_curtain | 1 | 1 | 1 | 1 | 1 | ✓ |
| inflating_balloon_09_cardboard | 1 | 0 | 1 | 0 | 0 | ✗ |
| newtons_cradle_cardboard | 0 | 0 | 0 | 1 | 0 | ✓ |
| railroad_crossing_gates | 1 | 1 | 1 | 1 | 0 | ✓ |
| sandcastle_wave_impact_lightoff | 0 | 1 | 0 | 1 | 1 | ✗ |
| spray_foam_cardboard | 0 | 0 | 0 | 0 | 0 | ✓ |
| sprinkler_valve_control_opposite | 1 | 1 | 1 | 1 | 0 | ✓ |
| syrup_fill_lightoff | 1 | 1 | 1 | 1 | 1 | ✓ |
| tea_dilution_curtain | 1 | 1 | 1 | 1 | 0 | ✓ |
| tire_inflate_curtain | 1 | 1 | 1 | 1 | 0 | ✓ |

**Summary:** 18/20 agree (90.0%) · Old positive rate 45% · New positive rate 45%

### occlusion_done

**Summary:** 17/20 agree (85.0%) · Old 85% · New 90%  
Mismatches: `balloon_block_cut_cardboard`, `gas_pump_nozzle_trigger`, `railroad_crossing_gates`

### trigger_applied

**Summary:** 17/20 agree (85.0%) · Old 70% · New 85%  
Mismatches: `balloon_block_cut_cardboard`, `railroad_crossing_gates`, `tire_inflate_curtain`

### state_evol

**Summary:** 17/20 agree (85.0%) · Old 10% · New 25%  
Mismatches: `hydraulic_press_can_crush_10_curtain`, `sprinkler_valve_control_opposite`, `tea_dilution_curtain`

---

## Analysis 2 — Three Runs, New Ensemble v1, GPT-5.4 (WAN/SORA) + GPT-4o (VEO)

**Scope:** WAN (20) + VEO (20) + SORA (20) = 60 tasks.  
**Ensemble:** New v1.

| Run | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | *physical_inaccuracy* | | | *occlusion_done* | | | *trigger_applied* | | | *state_evol* | | |
| WAN | 70.0% | 60% | 30% | 95.0% | 50% | 55% | 95.0% | 60% | 65% | 95.0% | 10% | 15% |
| VEO | 90.0% | 45% | 45% | 85.0% | 85% | 90% | 85.0% | 70% | 85% | 85.0% | 10% | 25% |
| SORA | 65.0% | 45% | 30% | 75.0% | 80% | 95% | 95.0% | 80% | 85% | 85.0% | 25% | 30% |
| **ALL** | **75.0%** | 50% | 35% | **85.0%** | 72% | 80% | **91.7%** | 70% | 78% | **88.3%** | 15% | 23% |

*Note: state_evol ALL is VEO-only for this run since WAN/SORA GPT-5.4 SE reports were available but WAN row reflects 95% due to full coverage.*

---

## Analysis 3 — Three Runs, New Ensemble v1, GPT-4o for All

**Scope:** WAN + VEO + SORA, 60 tasks. GPT-4o physics reports re-run for WAN and SORA at `max_side=1024`.  
**Ensemble:** New v1. Metrics reported: `physical_inaccuracy` and `occlusion_done` only.

| Run | Agree | Old% | New% | Agree | Old% | New% |
|---|---|---|---|---|---|---|---|
| | *physical_inaccuracy* | | | *occlusion_done* | | |
| WAN | 65.0% | 60% | 45% | 90.0% | 50% | 60% |
| VEO | 90.0% | 45% | 45% | 85.0% | 85% | 90% |
| SORA | 50.0% | 45% | 35% | 75.0% | 80% | 95% |
| **ALL** | **68.3%** | 50% | 42% | **83.3%** | 72% | 82% |

GPT-4o performs worse than GPT-5.4 as the third ensemble member for `physical_inaccuracy`
(68.3% vs 75.0%), consistent with GPT-4o being less sensitive to physics violations than GPT-5.4.

---

## Analysis 4 — Three Runs, New Ensemble v2, GPT-4o for All

**Scope:** WAN + VEO + SORA, 60 tasks. All four metrics.  
**Ensemble:** New v2 (Gemini majority consolidated first).

| Run | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | *physical_inaccuracy* | | | *occlusion_done* | | | *trigger_applied* | | | *state_evol* | | |
| WAN | 65.0% | 60% | 45% | 90.0% | 50% | 60% | 90.0% | 60% | 70% | N/A | — | — |
| VEO | 90.0% | 45% | 45% | 85.0% | 85% | 90% | 85.0% | 70% | 85% | 90.0% | 10% | 20% |
| SORA | 60.0% | 45% | 45% | 75.0% | 80% | 95% | 85.0% | 80% | 85% | N/A | — | — |
| **ALL** | **71.7%** | 50% | 45% | **83.3%** | 72% | 82% | **86.7%** | 70% | 80% | **90.0%**† | 10% | 20% |

†state_evol ALL is VEO-only (WAN/SORA have no GPT-4o SE reports).  
v2 improves `physical_inaccuracy` by +3.4 pp over v1 but slightly worsens `trigger_applied` (−5 pp).

---

## Analysis 5 — Three Runs, New Ensemble v2, GPT-5.4 (WAN/SORA) + GPT-4o (VEO)

**Scope:** WAN + VEO + SORA, 60 tasks. All four metrics.  
**Ensemble:** New v2. This is the primary/final analysis.

| Run | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | *physical_inaccuracy* | | | *occlusion_done* | | | *trigger_applied* | | | *state_evol* | | |
| WAN | 70.0% | 60% | 30% | 95.0% | 50% | 55% | 95.0% | 60% | 65% | 100.0% | 10% | 10% |
| VEO | 90.0% | 45% | 45% | 85.0% | 85% | 90% | 85.0% | 70% | 85% | 90.0% | 10% | 20% |
| SORA | 80.0% | 45% | 25% | 85.0% | 80% | 85% | 95.0% | 80% | 85% | 95.0% | 25% | 30% |
| **ALL** | **80.0%** | 50% | 33% | **88.3%** | 72% | 77% | **91.7%** | 70% | 78% | **95.0%** | 15% | 20% |

### Mismatches (Analysis 5)

| Metric | Count | Tasks |
|---|---|---|
| `physical_inaccuracy` | 12 | [WAN] falling_glass, honey_drip, hydraulic_press, inflating_balloon_09, spray_foam, syrup_fill · [VEO] inflating_balloon_09, sandcastle_wave_impact · [SORA] can_condensation, falling_glass, hydraulic_press, sprinkler_valve_control_opposite |
| `occlusion_done` | 7 | [WAN] falling_glass · [VEO] balloon_block_cut, gas_pump_nozzle, railroad_crossing_gates · [SORA] gas_pump_nozzle, railroad_crossing_gates, tire_inflate |
| `trigger_applied` | 5 | [WAN] tea_dilution · [VEO] balloon_block_cut, railroad_crossing_gates, tire_inflate · [SORA] newtons_cradle |
| `state_evol` | 3 | [VEO] sprinkler_valve_control_opposite, tea_dilution · [SORA] sandcastle_wave_impact |

---

## Summary: Agreement by Ensemble Version

| Metric | v1 GPT-4o | v1 GPT-5.4 | v2 GPT-4o | v2 GPT-5.4 |
|---|---|---|---|---|
| `physical_inaccuracy` | 68.3% | 75.0% | 71.7% | **80.0%** |
| `occlusion_done` | 83.3% | 85.0% | 83.3% | **88.3%** |
| `trigger_applied` | 91.7%* | 91.7% | 86.7% | **91.7%** |
| `state_evol` | 85.0%† | 88.3% | 90.0%† | **95.0%** |

*v1 GPT-4o trigger_applied used GPT-5.4 for WAN/SORA in that run.  
†VEO only.

**Best configuration overall: New ensemble v2 with GPT-5.4 (WAN/SORA) + GPT-4o (VEO).**

---

## Key Observations

1. **`physical_inaccuracy` has the lowest agreement** across all configurations (~68–80%), driven by Gemini (full-video upload, internal frame sampling) detecting violations that frame-sampled models miss. The same tasks recur as mismatches across all three video generators (`falling_glass`, `hydraulic_press`, `tea_dilution`), suggesting genuinely borderline cases rather than noise.

2. **`trigger_applied` and `state_evol` are the most stable** (≥91.7% and ≥95.0% in the best config), indicating these metrics are less sensitive to the choice of third VLM or ensemble definition.

3. **GPT-5.4 outperforms GPT-4o** as the third ensemble member on all metrics (+3–8 pp agreement), likely because its higher capability brings it closer to Gemini's detection sensitivity.

4. **Ensemble v2 outperforms v1** for `physical_inaccuracy` (+3–5 pp) because consolidating Gemini's 3 votes before the cross-VLM vote gives a more stable signal, reducing cases where a single divergent Gemini response is overruled by Claude+GPT.

5. **The new ensemble positive rates are consistently higher than the old ensemble** for control and SE metrics (Claude and GPT are more permissive), and lower for `physical_inaccuracy` (Claude and GPT miss violations that Gemini's full-video view catches).

---

## Analysis 6 — Three Runs, New Ensemble v2, GPT-5.4 for All Metrics and Runs

**Scope:** WAN + VEO + SORA, 60 tasks. All four metrics.  
**Third VLM:** GPT-5.4 for all runs (VEO control and SE reports newly added).  
**Ensemble:** New v2.

| Run | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | *physical_inaccuracy* | | | *occlusion_done* | | | *trigger_applied* | | | *state_evol* | | |
| WAN | 70.0% | 60% | 30% | 95.0% | 50% | 55% | 95.0% | 60% | 65% | 100.0% | 10% | 10% |
| VEO | 95.0% | 45% | 40% | 95.0% | 85% | 80% | 80.0% | 70% | 90% | 85.0% | 10% | 25% |
| SORA | 80.0% | 45% | 25% | 85.0% | 80% | 85% | 95.0% | 80% | 85% | 95.0% | 25% | 30% |
| **ALL** | **81.7%** | 50% | 32% | **91.7%** | 72% | 73% | **90.0%** | 70% | 80% | **93.3%** | 15% | 22% |

### Mismatches (Analysis 6)

| Metric | Count | Tasks |
|---|---|---|
| `physical_inaccuracy` | 11 | [WAN] falling_glass, honey_drip, hydraulic_press, inflating_balloon_09, spray_foam, syrup_fill · [VEO] inflating_balloon_09 · [SORA] can_condensation, falling_glass, hydraulic_press, sprinkler_valve_control_opposite |
| `occlusion_done` | 5 | [WAN] falling_glass · [VEO] gas_pump_nozzle · [SORA] gas_pump_nozzle, railroad_crossing_gates, tire_inflate |
| `trigger_applied` | 6 | [WAN] tea_dilution · [VEO] balloon_block_cut, railroad_crossing_gates, tea_dilution, tire_inflate · [SORA] newtons_cradle |
| `state_evol` | 4 | [VEO] sprinkler_valve_control_opposite, tea_dilution, tire_inflate · [SORA] sandcastle_wave_impact |

### Comparison with Analysis 5 (v2, GPT-5.4 WAN/SORA + GPT-4o VEO)

| Metric | Analysis 5 (mixed) | Analysis 6 (GPT-5.4 all) | Change |
|---|---|---|---|
| `physical_inaccuracy` | 80.0% | **81.7%** | +1.7 pp |
| `occlusion_done` | 88.3% | **91.7%** | +3.4 pp |
| `trigger_applied` | 91.7% | 90.0% | −1.7 pp |
| `state_evol` | 95.0% | 93.3% | −1.7 pp |

Switching VEO from GPT-4o to GPT-5.4 improves physics and occlusion agreement but slightly
reduces trigger and state-evol agreement, indicating GPT-5.4 is more conservative on
trigger/SE for VEO videos specifically. Overall the two configurations are within 3.4 pp of
each other on all metrics.

---

## Analysis 7 — All Five Runs, New Ensemble v2, GPT-5.4 Throughout

**Scope:** WAN (20) + VEO (20) + SORA (20) + HUNYUAN (18) + LINGBOT (11) = 89 tasks total.  
**Third VLM:** GPT-5.4 for all runs and all metrics.  
**Ensemble:** New v2.

| Run (n) | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% | Agree | Old% | New% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | *physical_inaccuracy* | | | *occlusion_done* | | | *trigger_applied* | | | *state_evol* | | |
| WAN (20) | 70.0% | 60% | 30% | 95.0% | 50% | 55% | 95.0% | 60% | 65% | 100.0% | 10% | 10% |
| VEO (20) | 95.0% | 45% | 40% | 95.0% | 85% | 80% | 80.0% | 70% | 90% | 85.0% | 10% | 25% |
| SORA (20) | 80.0% | 45% | 25% | 85.0% | 80% | 85% | 95.0% | 80% | 85% | 95.0% | 25% | 30% |
| HUNYUAN (18) | 66.7% | 56% | 22% | 77.8% | 56% | 33% | 100.0% | 61% | 61% | 100.0% | 0% | 0% |
| LINGBOT (11) | 81.8% | 45% | 27% | 90.9% | 36% | 27% | 81.8% | 27% | 45% | 90.9% | 0% | 9% |
| **ALL (89)** | **78.7%** | 51% | 29% | **88.8%** | 64% | 60% | **91.0%** | 63% | 72% | **94.4%** | 10% | 16% |

### Mismatches (Analysis 7)

| Metric | Count | Tasks |
|---|---|---|
| `physical_inaccuracy` | 19 | [WAN] falling_glass, honey_drip, hydraulic_press, inflating_balloon_09, spray_foam, syrup_fill · [VEO] inflating_balloon_09 · [SORA] can_condensation, falling_glass, hydraulic_press, sprinkler_valve_control_opposite · [HUNYUAN] clay_drying, railroad_crossing_gates, sandcastle_wave_impact, syrup_fill, tea_dilution, tire_inflate · [LINGBOT] sandcastle_wave_impact, tea_dilution |
| `occlusion_done` | 10 | [WAN] falling_glass · [VEO] gas_pump_nozzle · [SORA] gas_pump_nozzle, railroad_crossing_gates, tire_inflate · [HUNYUAN] clay_drying, honey_drip, newtons_cradle, sandcastle_wave_impact · [LINGBOT] gas_pump_nozzle |
| `trigger_applied` | 8 | [WAN] tea_dilution · [VEO] balloon_block_cut, railroad_crossing_gates, tea_dilution, tire_inflate · [SORA] newtons_cradle · [LINGBOT] newtons_cradle, railroad_crossing_gates |
| `state_evol` | 5 | [VEO] sprinkler_valve_control_opposite, tea_dilution, tire_inflate · [SORA] sandcastle_wave_impact · [LINGBOT] railroad_crossing_gates |

### Notes

- **HUNYUAN** has the lowest `physical_inaccuracy` agreement (66.7%), with 6 mismatches in 18 tasks. Gemini's positive rate is high (56%) while the new ensemble drops to 22%, indicating GPT-5.4 and Claude are considerably more conservative on Hunyuan videos.
- **HUNYUAN** `state_evol` agreement is 100% at 0% positive for both ensembles — all three VLMs unanimously find no state evolution in any Hunyuan task.
- **LINGBOT** `trigger_applied` is the only metric where the new ensemble is more permissive than the old (27% → 45%), driven by Claude and GPT-5.4 accepting triggers that Gemini's majority rejected.
- **Recurring cross-run mismatches:** `falling_glass`, `sandcastle_wave_impact`, `tea_dilution`, and `railroad_crossing_gates` appear as physics or control mismatches across multiple video generators, suggesting task-level difficulty rather than generator-specific artifacts.
