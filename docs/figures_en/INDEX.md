# English figures: index

English replacements for the pictures of the base deck
`Presentacion_MLSMFP_05-10_resultados_fase_c2.pptx` that contain Spanish text.
The images are not inserted into any slide. Generator scripts and commands:
`scripts/figures_en/README.md`. All files have the same pixel size (and
therefore aspect ratio) as the original pictures.

| Slide | Original picture (hash) | New file | How generated | Fidelity notes / limitations |
|---|---|---|---|---|
| 2 | `Imagen 7` (`3c289b84`, 6734x3118) | `slide02_gantt_schedule_en.png` | `gantt_schedule.py`, redrawn with matplotlib from a data block transcribed from the original | All 15 bars, the 4 milestones and the 2 deadlines were measured against the week grid and match the original (see table below). The label panel was widened to the left, into the empty margin of the original, because the English task labels are longer and would otherwise run into the bars. **Flag:** the original red label is cut off by the header band ("ENTREGA SOLICITU..."); read as "ENTREGA SOLICITUD 20 NOV 2026" and translated as "REQUEST SUBMISSION 20 NOV 2026", split over three lines so that it does not reach the header band. As in the original, the rotated milestone and deadline labels overlap the week labels. |
| 3 and 4 | `Imagen 2` / `Imagen 4` (`8726ebb0`, 2340x2080, same image) | `slide03_hydrograph_shapes_en.png` | `hydrograph_panels.py` from `data/hydrograph_shapes/*.csv` | Same 18 panels and order: 8 existing shapes in blue (`common_bogota`, `common_storm`, `long_moderate`, `new_hydrograph`, `new_hydrograph_2`, `rare_extreme`, `shape_2h`, `shape_5h`) and 10 new in orange. All panels reproduced from the CSVs; curves visually identical to the original. |
| 6 | `Imagen 2` (`0c672324`, 2172x782) | `slide06_new_training_shapes_en.png` | `hydrograph_panels.py` from `data/hydrograph_shapes/*.csv` | The 10 shapes of this panel are the newest batch (`shape_micro_burst_45min` ... `shape_twin_plateau_15h`), not the 10 orange shapes of slide 3. dur / t_peak recomputed from the data: all 10 match the original annotations. |
| 11 | `Imagen 5` (`1513342d`, 1790x782) | `slide11_validation_shapes_en.png` | `hydrograph_panels.py` from `data/hydrograph_validation/*.csv` | 5 interpolation (green) + 2 extrapolation (red), same order. dur / t_peak recomputed: all 7 match the original annotations. |
| 15 | `Imagen 4` (`a4fa09e4`, 2136x2721) | `slide15_flowchart_training_en.png` | `flowcharts.py`, matplotlib patches at positions measured on the original | Same layout, shapes and connectors. "Chicó Sur" kept (place name). The rotated side note was moved slightly down so that it no longer touches the "Study network" box (it overlapped the box in the original). |
| 15 | `Imagen 12` (`597a5d10`, 2136x1965) | `slide15_flowchart_validation_en.png` | `flowcharts.py` | Same layout, shapes and connectors. |

## Gantt transcription (slide 2)

Weeks W1-W22 start on Monday 10-aug-2026. Bars are inclusive week ranges.

| Section | Task | Weeks |
|---|---|---|
| Conceptual consolidation | Conceptual clarity and adjustments to the observations | W1-W9 |
| | Literature review and methodological justification | W1-W9 |
| | Research on new models and spatio-temporal variables | W4-W7 |
| Experimental design | Design of scenarios, new variables and automation | W2-W4 |
| | Design and preliminary tests of neural networks | W6-W11 |
| Iterative experimentation | Pilot batch · quality control and baseline | W3-W4 |
| | Cycle 1 · +1,000 hydrographs → retraining | W4-W6 |
| | Cycle 2 · +1,000 hydrographs → retraining | W6-W8 |
| | Cycle 3 · +1,000 hydrographs → sufficiency analysis | W8-W10 |
| Modeling and validation | Final optimization and architecture comparison | W10-W13 |
| | External validation, interpretation and technical discussion | W12-W15 |
| Document and closure | Progressive writing and updating of the document | W1-W15 |
| | Comprehensive review and final corrections | W14-W18 |
| | Thesis defense preparation and rehearsal | W19-W22 |
| Complementary (hatched) | Evaluation of an uncalibrated network, subject to progress | W14-W15 |

Milestones (dashed, at the end of the week): Retraining 1 = end of W6
(20-sep), Retraining 2 = end of W8 (04-oct), Data sufficiency = end of W10
(18-oct), Final model = end of W13 (08-nov). Deadlines (red): request
submission 20-nov-2026 (W15), defense 08-jan-2027 (W22). Curved arrows link
Cycle 1 → Cycle 2 → Cycle 3.

## Other pictures of the base deck

| Slide | Picture (hash) | Spanish text? | Action |
|---|---|---|---|
| 5 | `Imagen 5` / `Imagen 7` (`8451b243`, same image) | Yes, title only: "Formas de hidrograma nuevas (8) - 2026-09-01" | **Not regenerated.** The 8 shapes shown (ultra short flash, extended frontal storm, quad peak irregular, bimodal wide separation, plateau sustained, stepped recession, very late extreme peak, sawtooth bursts) are not in any versioned data file or script. Panel titles and axis labels are already English. |
| 7, 9, 30 | flood maps (`060d1d53`, `3ad10389`, `a2b25a17`, `00a0e676`, `b10f673c`, `5cd46ba4`) | No | none |
| 8, 10, 30 | parity plots / inflow hydrograph (`b5eeadde`, `4823333b`, `aa3faef5`, `cfc5bed4`) | No | none |
| 16 | `Imagen 6` (`2f75b044`, .wmf table, node 75 sample) | No (English column headers; values and shape ids are identifiers) | left as is |
| 34 | parity plots `fa3af9bd` (Linear), `e8894212` (MLP), `b6b853b4` (XGBoost), `747f0306` (Random Forest) | Only the run identifier in the title: "Node-Level Parity - todas_las_corridas" | **Not regenerated.** Produced by `scripts/generar_paridad_por_familia.py`, which needs a full out-of-fold re-evaluation of each family (predictions are not cached) and, after the audit fixes, would not reproduce the numbers on the slide. New parity figures for the Part 2 models are produced separately. |
