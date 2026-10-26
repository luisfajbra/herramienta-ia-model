# Study guide for the advisor presentation (audit, approximate SVM and fair model comparison)

- **Prepared:** 2026-10-20, branch `worktree-optuna-tuning-familia-ganadora` (HEAD `86cec4d` when this guide was written).
- **For:** the thesis author, before presenting the deck `Presentacion_MLSMFP_20-10_avances_auditoria_svm.pptx` (53 slides) to the advisor.
- **Rule used throughout:** every number cites its source file (and column where useful). Text marked **Interpretation** is a reading of the numbers, not a measurement. Text marked **Not verified** was not measured by anyone in the project.
- **Source abbreviations** (all paths relative to the worktree root):

| Abbreviation | File |
|---|---|
| R04 | `docs/auditoria/correcciones/04-informe-consolidado.md` (consolidated report, Part 1) |
| R01, R02, R03, R05, R06 | `docs/auditoria/correcciones/01-…`, `02-…`, `03-…`, `05-…`, `06-…` |
| AUD | `docs/auditoria/2026-10-10-auditoria-completa.md` (original audit) |
| A01 … A05 | `docs/auditoria/anexos/01-…` … `05-…` (audit annexes) |
| CP1, MC1, BL1, PF1 | `docs/auditoria/correcciones/datos/{comparacion_protocolos, matriz_confusion, banda_limitrofe, por_forma}.csv` (Part 1, XGBoost with default parameters) |
| CP, MC, BL, PF, RA, CA, TP | `docs/auditoria/correcciones/datos_parte2/{comparacion_protocolos, matriz_confusion, banda_limitrofe, por_forma, resumen_afinacion, convergencia_afinacion}.csv` and `tables_provenance.json` (Part 2, final run) |
| FIG | `docs/figures_part2/INDEX.md` and `figure_numbers.json` |
| SPEC2 | `docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md` |
| INF0927 | `../../../INFORME_ULTIMOS_CAMBIOS_PARA_ASESORIA.md` (study report of the state at 2026-09-27, in the main checkout) |

## Table of contents

0. [How to use this guide, 60-second pitch and 5-minute story](#0-how-to-use-this-guide)
1. [Background in plain words](#1-background-in-plain-words)
2. [Evaluation protocols: the single most important concept](#2-evaluation-protocols-the-single-most-important-concept)
3. [What happened since the last meeting](#3-what-happened-since-the-last-meeting)
4. [The audit explained, finding by finding](#4-the-audit-explained-finding-by-finding)
5. [The label resolution spike (D-04 / H-06)](#5-the-label-resolution-spike-d-04--h-06)
6. [The approximate SVM](#6-the-approximate-svm)
7. [The fair comparison: XGBoost, MLP and approximate SVM](#7-the-fair-comparison-xgboost-mlp-and-approximate-svm)
8. [Limits and what NOT to claim](#8-limits-and-what-not-to-claim)
9. [Q&A bank](#9-qa-bank)
10. [Slide-by-slide walkthrough](#10-slide-by-slide-walkthrough)
11. [Numbers cheat sheet](#11-numbers-cheat-sheet)
12. [Glossary English and Spanish, and speaking tips](#12-glossary-english-and-spanish-and-speaking-tips)
13. [Where everything is, repository state and ready-to-present checklist](#13-where-everything-is-repository-state-and-ready-to-present-checklist)

---

## 0. How to use this guide

**Study order (about 3 hours).** Read §0 and §2 twice: §2 (protocols) is what the advisor is most likely to probe. Then §7 (results), §4 (audit), §5 (label), §6 (SVM). Use §9 (Q&A) as flash cards and §11 (cheat sheet) the morning of the meeting. §10 is your speaker script, slide by slide.

**Three rules for the meeting.**

1. Always say **which protocol** a number comes from. A number without its protocol is meaningless in this project.
2. Separate **measured** from **interpreted**. Say "we measured…" for numbers and "a plausible explanation is…" for causes nobody tested.
3. When unsure, say what was measured and what was not. Honesty about limits is a strength in front of an advisor, not a weakness.

### 0.1 The 60-second pitch

> "We build a machine-learning surrogate of SWMM that predicts, for each of the 160 nodes of the Chicó Sur network, whether it floods and how much volume overflows. An internal audit on October 10 showed that two of our three validation protocols only measured interpolation between load factors: a table that just interpolates beats the model there. We corrected the evaluation, added two no-learning baselines and made 'leave one whole hydrograph shape out' the main test. Under that test, and with the same data, the same protocol and the same tuning budget for three model families, tuned XGBoost has the lowest pooled volume error, 69.64 m³ against 172.86 m³ for the best no-learning baseline; the MLP is second with 101.12 m³; an approximate SVM falls below the baseline. Classification is almost perfect for XGBoost and the MLP (F1 about 0.98). The weak spot is the volume of long, plateau-like hydrographs, and these figures are an exploratory estimate on a single network."

Sources: CP (`rmse_vol_agregado_m3`, `f1_agregado`); R04 §1; R06 §1.

### 0.2 The 5-minute story

1. **Problem.** SWMM is accurate but each scenario must be simulated. A surrogate trained on SWMM runs could answer "which nodes flood and how much" instantly. To trust it, it must be tested on scenarios it never saw.
2. **What we found (audit, 2026-10-10).** 111 findings, 2 critical (AUD §4). Critical 1 (M-01): the protocols LOSO and GroupKFold5 hold out load factors; with 25 closely spaced factors, interpolating the neighbouring factors of the same node and shape gives a volume RMSE of 9.13 m³ against 57.89 m³ for XGBoost (CP1). So those protocols do not prove learning. Critical 2 (M-02): with unseen hydrograph shapes, the volume generalizes worse; one shape dominates the error. Also: F1 had been averaged per fold with zeros (M-03), which reported 0.7379 instead of 0.9667.
3. **What we fixed (2026-10-11).** Two no-learning baselines; LeaveOneShapeOut as main protocol; pooled metrics and the confusion matrix; out-of-fold predictions saved with SHA-256 fingerprints; in-sample figures labelled; model provenance checked before predicting; a dangerous script neutralised. A spike showed the flood label is a deterministic quantization (effective threshold ≈ 0.5 m³ of continuous volume), not noise.
4. **What the models do now (2026-10-13 run).** Same data (154,080 node-scenario rows), same protocol (40 folds, one shape out), same budget (20 Optuna trials). Tuned XGBoost: pooled RMSE 69.64 m³, F1 0.9812, AUC 0.9993. Tuned MLP: 101.12 m³, F1 0.9783. Approximate SVM: 214.87 m³, worse than the baseline (172.86 m³). At scenario level the MLP has a slightly lower median error (12.7 % against 15.5 %), so XGBoost does not win every metric.
5. **What is still open.** The volume of plateau-like shapes (`shape_quasi_uniform_block` is underestimated by 31.2 % by tuned XGBoost and carries 61.0 % of its squared error); hyperparameters were chosen with all shapes (optimistic, equally for the three); one seed; one network; the label decision (path A) must be confirmed; nested tuning; the 7 reserved validation shapes; older figures not regenerated.

---

## 1. Background in plain words

### 1.1 What the surrogate is

A **surrogate** (Spanish: modelo sustituto) is a fast approximation trained on the outputs of a slower, physically based model. Here the slow model is **SWMM 5.2** (EPA Storm Water Management Model), run with **dynamic wave** routing (`DYNWAVE`), routing step 5 s and report step 3 min (R03 header; A01 header). The surrogate is **two-stage** (R06 §2):

1. a **classifier** answers "does this node flood in this scenario?" (yes/no);
2. a **regressor** answers "how many m³ overflow?";
3. the **end-to-end** prediction is the regressor volume where the classifier says "flood", and 0 elsewhere. All volume metrics in this guide are end-to-end over all nodes unless they say "oracle".

### 1.2 The network

- `data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp` (Chicó Sur): **160 junctions**, 1 free outfall (`109C`), 160 circular PVC conduits (Manning n = 0.0082), flow units L/s, `ALLOW_PONDING NO` (A01 header). It is an acyclic tree (A01, H-12).
- No subcatchments or rain gauges: inflows are hydrographs imposed directly at the nodes (deck slide 18).

### 1.3 The scenarios

- **Hydrograph shapes:** 40 (the `base` shape of the `.inp` plus 39 CSV shapes, each a normalized curve `time_h, q_norm` with peak 1) (A01 header; INF0927 §3.1).
- **Load factors** (`factor_mult`): 25 values, 0.2 to 5.0 in steps of 0.2; each node's base inflow is multiplied by the factor (docs/FLUJO_ACTUAL.md §1).
- **Grid:** 40 × 25 = **1,000 SWMM runs**. Runs with |continuity error| > 5 % are excluded: 37 runs, leaving **963** (TP `n_runs`, `n_missing_cells`). The excluded cells concentrate in short or extreme shapes: `shape_ultra_short_impulse` 10, `rare_extreme` 7, `common_bogota` 5 (R04 §5.3).
- **Rows:** 963 runs × 160 nodes = **154,080 node-scenario rows**, of which **46,225** are flooded (R03 §2.2; MC `fn + tp`).

### 1.4 Features (contract `tabular_v4_21`, 21 inputs)

A **feature contract** is the frozen, ordered list of inputs a model expects. `tabular_v4_21` = the 17 inputs of `tabular_v3_17` minus `duracion_horas` and `tiempo_al_pico_h`, plus 6 tributary features (`swmm_resilience/ml/contracts.py`, `FEATURE_DEFINITIONS_V21`). Full list from `datos/runs/provenance.json`:

| Group | Features | Plain meaning |
|---|---|---|
| Static node and pipe geometry | `elev_fondo`, `prof_max`, `n_tuberias_in`, `n_tuberias_out`, `diam_max_in`, `diam_max_out`, `pendiente_max_in`, `pendiente_out`, `dist_outfall_m` | invert elevation, max depth, number and size of in/out pipes, slopes, distance to the outfall |
| Topology and capacity | `n_nodos_aguas_arriba`, `upstream_capacity_lps`, `base_inflow_lps`, `q_pico_acum_base` | upstream node count, capacity of the upstream conduits, base inflow, sum of upstream base peaks |
| Scaled peaks | `q_pico_nodo`, `q_pico_acum_escalado` | the node's peak and the accumulated upstream peak after applying the factor |
| **Volume / tributary (new in v4_21)** | `volumen_local_integrado_m3`, `volumen_tributario_integrado_m3`, `duracion_efectiva_tributario_h`, `tiempo_primer_pico_tributario_h`, `tiempo_sobre_capacidad_h`, `integral_exceso_capacidad_m3` | built from **Q_tributario(t)**, the sum over time of the node's own inflow series and those of all its upstream nodes: its integrated volume, how long it stays above 1 % of its peak, time to its first peak, how long it exceeds the upstream capacity, and the volume above that capacity (`swmm_resilience/extraction/tributary_features.py`) |

**Intuition.** Peak-only features cannot tell a 30-minute burst from a 14-hour plateau with the same peak, yet the flooded volume differs enormously (the audit measured up to 13.3 times at equal duration and factor; A01, H-02). The tributary features give the model "how much water arrives and for how long above capacity", which is what a hydraulic engineer would look at.

**Controlled evidence that they help:** on the same rows and protocol (LeaveOneShapeOut), per-shape mean RMSE 78.0368 m³ with `tabular_v3_17` against 56.8088 m³ with `tabular_v4_21` (`docs/resultados/2026-10-06-explicacion-completa.md` §11.1; reproduced as 56.81 in CP1 `rmse_vol_media_por_fold_m3`).

### 1.5 Targets

- `inunda` (flood label, 0/1): 1 if the volume reported by SWMM in the `.rpt` is ≥ 1 m³ (TP `flood_threshold_m3`). §5 explains why this is effectively ≈ 0.5 m³ of continuous volume.
- `vol_inundacion_m3` (flood volume, m³): *Total Flood Volume* of the `.rpt` × 1000, integer resolution (R03 §2.1).

### 1.6 Models compared

| Family | Plain description |
|---|---|
| XGBoost | gradient-boosted decision trees: many small trees, each correcting the previous ones |
| MLP (multilayer perceptron) | a dense neural network with 1 to 3 hidden layers |
| Approximate SVM | support vector machine whose RBF kernel is approximated with a Nystroem map (§6) |

Earlier material (deck slide 33, 2026-10-05) also had Random Forest and a linear model; they were not part of the fair comparison.

### 1.7 Background up to September 27 (from INF0927) and what is superseded

Still valid, worth knowing:

- **Shape catalogue.** Commit `6e8579b` (2026-09-27) added 21 training shapes in `data/hydrograph_shapes/` and **7 reserved validation shapes** in `data/hydrograph_validation/` (5 labelled interpolation, 2 extrapolation by duration; deck slide 11 and `docs/figures_en/INDEX.md`).
- **Shape features in inference.** Before that commit, `predict_network` set `duracion_horas` and `tiempo_al_pico_h` to 0 at prediction time while training used real values; the fix computes both from the `.inp` time series (`base_shape_stats()` in `swmm_resilience/ml/predict.py`). The "up to 94 %" underestimation is a motivation quoted in the commit message, not a measured result (INF0927 §4.1).
- **Command.** `python main.py --evaluate-validation-shapes data/hydrograph_validation` expands each reserved shape to scenarios (by default the 24 midpoints 0.3 … 4.9, up to 168 scenarios) and refuses shapes whose name also exists in the training catalogue (INF0927 §3.2–3.3). The interpolation/extrapolation label refers only to duration.
- **Reproducibility.** Commit `9d6ec4a` sorted the training frame (`shape_id`, `factor_mult`, `node_id`) because XGBoost row subsampling depends on row order (INF0927 §6.1).

Superseded by later work:

| INF0927 statement | Current state |
|---|---|
| "No results exist for generalization to new shapes" | Generalization to unseen shapes is now measured with LeaveOneShapeOut over the 40 training shapes (CP, CP1). **The 7 reserved validation shapes still have no results table in the sources reviewed**: that independent test remains pending, and the audit noted that `--evaluate-validation-shapes` protects the out-of-sample condition only by shape name (A02, M-13). |
| "The bench has no final ranking" | A fair 3-family comparison exists (R06, 2026-10-13); an earlier 5-family ranking was presented on slide 33 (2026-10-05). |
| `duracion_horas` / `tiempo_al_pico_h` are model inputs | They are **not** inputs of `tabular_v4_21`; shape information now enters through the 6 tributary features. The 09-27 fix still matters for the legacy `tabular_v3_17` path (`python main.py` still trains v3_17, README.md note). |
| "The models in `outputs/models` were retrained" | The active model is the XGBoost `tabular_v4_21` promoted on 2026-10-06 (`promotion.json`, source `scripts/comparar_contratos_xgboost.py`), verified against its SHA-256 on 2026-10-11 (R02 §3). It is **not** the Part 2 tuned model; no Part 2 model was promoted (SPEC2 §3.4). |

**What to say:** "Since September we moved from 'we built the tools to test new shapes' to 'we measured generalization to unseen shapes under a protocol that a no-learning baseline cannot game'; the 7 reserved shapes are the next, independent check."

---

## 2. Evaluation protocols: the single most important concept

### 2.1 Cross-validation in one paragraph

Cross-validation splits the data into **folds**. In each fold the model is trained on part of the data and predicts the held-out part. Collecting the held-out predictions of all folds gives **out-of-fold (OOF)** predictions: one honest prediction per row, made by a model that never saw that row. What matters is **what is held out together** (the "group"), because rows that are very similar to training rows make the test easy.

### 2.2 The three protocols

| Protocol | Group held out | Folds | What it really measures | Source |
|---|---|---|---|---|
| **LOSO** | one complete load factor (e.g. all rows with factor 3.0) | 25 | interpolation between factors: every shape is still in training at ±0.2 | R01 §0; A02 §1.3 |
| **GroupKFold5** | five factors per fold | 5 | the same interpolation; with this grid no fold holds out two contiguous factors | R01 §1 reading 2 |
| **LeaveOneShapeOut** | one complete hydrograph shape (all 25 factors) | 40 | generalization to an unseen hydrograph shape. **Main protocol** | R01 §0 |

Naming note: "LOSO" groups by factor although older documents spelled it "Leave One Scenario Out"; the audit recommends calling it "LeaveOneFactorOut" in the thesis (A02, line 213).

**Why GroupKFold5 is not harder than LOSO here.** The folds interleave factors (fold 0 = {0.6, 2.0, 3.0, 4.0, 5.0}, fold 1 = {0.4, 1.8, 2.8, 3.8, 4.8}, …), so every held-out factor keeps both neighbours in training. The interpolation baseline gives exactly the same predictions in both protocols (maximum difference 0.0 over 154,080 rows; R01 §1).

### 2.3 Why a no-learning baseline matters

A **baseline** is a predictor that uses no learning. If the model cannot beat it, the protocol (or the model) shows nothing. Two baselines were added (R01 §1; `swmm_resilience/ml/bench/baselines.py`):

| Baseline | Used in | Rule |
|---|---|---|
| Factor interpolation (`baseline_interpolacion_factor`) | LOSO, GroupKFold5 | for (shape, node, held-out factor), linearly interpolate the volume of the same node and shape between the neighbouring training factors |
| Nearest shape (`baseline_forma_mas_cercana`) | LeaveOneShapeOut | copy the volume of the same node and factor from the training shape most similar in (duration, time to peak), standardized |

Both predict the label as `volume ≥ 1 m³`. They do not produce probabilities, so their ROC-AUC is not interpreted (R01 §1). Fallbacks caused by the 37 missing cells: 1.25 % of interpolation rows and 0.42 % of nearest-shape rows (R04 §3).

### 2.4 Worked toy example: interpolation versus a new shape

Imagine node N under shape S. SWMM gives 100 m³ at factor 2.8 and 140 m³ at factor 3.2.

- **LOSO holds out factor 3.0.** The baseline says (100 + 140) / 2 = 120 m³. SWMM probably says about 120 m³. Nothing about hydraulics was learned: the answer was "between the neighbours".
- **LeaveOneShapeOut holds out shape S completely.** There is no S at 2.8 or 3.2 to interpolate. The nearest-shape baseline must borrow from another shape and can be badly wrong: `shape_quasi_uniform_block` and `shape_slow_broad_rise` choose each other as neighbours and swap volumes (1,767,048 and 684,492 m³), giving total volume errors of −61.3 % and +158.2 % (R01 §1, remaining limits; PF1).

Real numbers (CP1, XGBoost with default parameters, 154,080 rows):

| Protocol | Model | Pooled RMSE (m³) | Pooled F1 |
|---|---|---|---|
| LOSO | factor interpolation (no learning) | 9.13 | 0.9718 |
| LOSO | XGBoost | 57.89 | 0.9667 |
| LeaveOneShapeOut | nearest shape (no learning) | 172.86 | 0.9764 |
| LeaveOneShapeOut | XGBoost | 106.21 | 0.9817 |

In LOSO the table wins; in LeaveOneShapeOut the model wins. That reversal is the evidence that LeaveOneShapeOut is the meaningful test.

### 2.5 Pooled versus per-fold metrics

- **Pooled:** concatenate all OOF predictions of the protocol and compute one metric (one confusion matrix, one RMSE).
- **Per-fold mean:** compute the metric in each fold and average.

**Toy example (F1 with an empty fold).** Five folds; in fold 1 there is no flooded node and the model correctly predicts none. F1 = 2TP / (2TP + FP + FN) = 0/0 there: undefined. The old code replaced undefined by 0 (`zero_division=0`). Folds 2–5 have F1 = 1. Zero-filled mean = (0 + 1 + 1 + 1 + 1) / 5 = **0.80**; pooled F1 = **1.00**. A perfect model scored 0.80. This is exactly what happened in LOSO: 5 of 25 factors (low factors) have no flooded node, and a perfect predictor got 0.8000 (R01 §3, Table 3.1).

For RMSE both definitions are legitimate but answer different questions: the **pooled RMSE** weights each shape by its squared error (dominated by large-volume shapes), the **per-fold mean** weights each shape equally (R06 §3.1). In this project the pooled RMSE is the main metric (R06 §2).

### 2.6 Other metric words, explained once

| Term | Meaning here |
|---|---|
| TP / FP / FN / TN | true positive (flooded, predicted flooded) / false positive (dry, predicted flooded: false alarm) / false negative (flooded, predicted dry: missed flood) / true negative |
| Recall | TP / (TP + FN): share of real floods found |
| Precision | TP / (TP + FP): share of alarms that are real |
| F1 | harmonic mean of precision and recall |
| Specificity | TN / (TN + FP) |
| ROC-AUC (AUC) | probability that a random flooded row gets a higher flood probability than a random dry row; independent of the 0.5 threshold |
| RMSE | root mean squared error of volume, m³ |
| NSE | Nash–Sutcliffe efficiency, 1 − SSE / Σ(obs − mean)²; 1 is perfect |
| Oracle NSE | NSE of the regressor alone on the rows that really flooded (classifier errors removed) |
| Total volume error | (Σ predicted − Σ SWMM) / Σ SWMM |

**What to say:** "LOSO and GroupKFold5 answer 'can you fill a gap between two load factors you already saw?'; a lookup table does that better than any model. LeaveOneShapeOut answers 'can you handle a hydrograph shape you never saw?', and that is the question of the thesis."

---

## 3. What happened since the last meeting

| Date | Step | Why |
|---|---|---|
| 2026-09-27 | Commit `6e8579b`: shape features in inference, 7 reserved validation shapes, `--evaluate-validation-shapes` (INF0927) | coherence between training and inference; enable out-of-sample tests |
| 2026-10-05 | First real end-to-end run: 1,000 SWMM runs, 0 failures; 37 runs over 5 % continuity error; 5-family benchmark; exact SVM cancelled after more than 1 day 15 h of CPU (deck slides 31–33; SPEC2 §3.1) | produce the full dataset and a first ranking |
| 2026-10-06 | Controlled contract comparison (78.0368 → 56.8088 m³) and promotion of XGBoost `tabular_v4_21` (`docs/resultados/2026-10-06-explicacion-completa.md` §11) | decide whether the tributary features are worth it |
| 2026-10-10 | Full audit: 5 parallel auditors, 111 findings (2 critical, 25 major, 48 moderate, 36 minor) (AUD §2, §4) | check before citing any result in the thesis |
| 2026-10-11 | Part 1 corrections (steps 1–6), spike on the label (D-04/H-06), reports R01–R05, decision log R1–R14 (R05) | fix evaluation, traceability and safety |
| 2026-10-12 | Author's decisions: label path A; XGBoost and MLP always presented; SVM as third candidate (SPEC2 header). Approximate SVM implemented (`52f3937`, `5a961da`), fold parallelism (`99d8cc0`), resumable run script (`8a06605`) | make the SVM feasible and the comparison fair |
| 2026-10-13 | The three families ran at the same time on a 64-CPU machine (start 02:55 UTC); tables built 17:19 UTC; report R06; deck of 13-10 (47 slides); English translation of documents and messages (`6c68984`, `b2dd4c9`, i18n commits) | obtain comparable results and prepare an English presentation |
| 2026-10-20 | English versions of the Spanish pictures of the base deck (`beb4c19`); parity plots, error by shape and flood maps for the tuned models (`86cec4d`); new deck of 20-10 (53 slides) | the presentation is in English; show spatial and scenario-level behaviour, not only pooled numbers |

**What to say:** "The work since the last meeting is one line: audit, correct the evaluation, make the SVM feasible, compare three families fairly, and show the results as maps and parity plots."

---

## 4. The audit explained, finding by finding

**Context.** The audit (AUD) was read-only, ran about 150 control SWMM simulations, re-generated XGBoost OOF predictions and verified its critical findings by a second route (AUD §2–3). Its verdict: **the hydraulic data chain is sound** (re-simulations reproduce the DB to the m³ in 11 shape/factor combinations; rescaling preserves volume within 0.08 %); **the problem was what was concluded from the data** (AUD §1). Severity scale: 🔴 critical, 🟠 major, 🟡 moderate, 🔵 minor.

### 4.1 M-01 (🔴): LOSO and GroupKFold5 measure interpolation; no baselines

- **Plain words:** the main tests could be won by a lookup table.
- **Why it matters:** any claim "the model generalizes" based on LOSO is unsupported.
- **Changed:** two baselines in the benchmark (`9c8db1b` and follow-ups); `PROTOCOL_INTERPRETATION` written into every report; LeaveOneShapeOut declared main (R01 §1).
- **Before → after:** interpolation 9.00 vs XGBoost 68.43 m³ (`tabular_v3_17`, 160,000 rows) → 9.13 vs 57.89 m³ (`tabular_v4_21`, 154,080 rows) (R04 §2). Not a controlled comparison: rows and contract changed together.
- **What to say:** "We kept LOSO only as an interpolation reference; every generalization claim now comes from LeaveOneShapeOut with a baseline next to it."

### 4.2 M-02 (🔴): volume generalizes poorly to new shapes

- **Plain words:** the model knows *where* it floods better than *how much* for shapes it never saw.
- **Why it matters:** flood volume is the hydraulic quantity of interest.
- **Changed:** this series measured it (per-shape breakdown, pooled RMSE, total volume error); the model and features were not changed (R01 §2). The `tabular_v4_21` contract that adds volume features predates the series.
- **Before → after:** oracle NSE 0.8141 → 0.8906; per-shape total volume error −63.3 %…+216.3 % → −52.1 %…+33.3 %; share of squared error of the worst shape 61.35 % → 71.6 % (it **rose**) (R04 §2). With tuning (Part 2): oracle NSE 0.9530, worst shape share 61.0 % (CP; R06 §4.3).
- **Status:** open (R04 §1).
- **What to say:** "Volume features helped, tuning helped more, but plateau-like shapes are still underestimated and dominate the error."

### 4.3 M-03 (🟠): F1 averaged per fold with zeros

- **Plain words:** see the toy example in §2.5.
- **Changed:** `classifier` in the benchmark is now pooled; the old value is kept under the explicit name `classifier_legacy_mean_zero_fill`; per-fold dispersion is reported with NaN for undefined folds (R01 §3).
- **Before → after:** LOSO F1 published 0.7408 (pooled 0.9656) → zero-filled 0.7379 vs pooled 0.9667 (CP1). The apparent LOSO vs GroupKFold5 gap (0.7379 vs 0.9672) disappears with pooling (0.9667 vs 0.9681) (R01 Table 3.3).
- **Still open:** `by_factor` and the legacy evaluator `ml/evaluator.py` still average with zeros; do not quote `outputs/metrics/metrics_classifier.json` (F1 0.7408) (R01 §3).
- **What to say:** "The low LOSO F1 in older slides was an aggregation artefact: a perfect predictor would have scored 0.80."

### 4.4 M-08 / M-10 / T-19 (🟡): OOF predictions persisted, confusion matrix

- **Plain words:** before, the benchmark threw away the per-row predictions, so no confusion matrix could be rebuilt.
- **Changed:** OOF predictions saved per row as parquet with `prep_id`, hyperparameters and SHA-256 fingerprints; confusion matrix, specificity, per-shape breakdown, MAE and total volume error added (R01 §4). MCC, balanced accuracy and PR-AUC are still missing (R04 §1).
- **Numbers:** XGBoost default, LeaveOneShapeOut: TN 106,526 / FP 1,329 / FN 380 / TP 45,845; recall 0.9918, specificity 0.9877 (MC1; R01 Table 4.1).
- **What to say:** "Every table can be rebuilt from saved predictions whose fingerprints are recorded; nothing has to be retrained to check a number."

### 4.5 V-01 (🟠, and M-13 in part): in-sample figures without a label

- **Plain words:** resilience curves, total-volume curves and per-factor comparisons were computed on the same shapes, factors and nodes used for training, but looked like validation.
- **Changed:** an "In-sample" footnote on 5 families of ML figures and a console banner in `--evaluate-shapes` (`93e34b4`, 6 tests) (R02 §1).
- **Not done:** existing figures were not regenerated; `--evaluate-generalization` and `--evaluate-validation-shapes` are not labelled (R02 §1).
- **What to say:** "Figures that show fit to training data now say so; the out-of-sample evidence is the LeaveOneShapeOut predictions."

### 4.6 V-07 (🟠): a script could overwrite the base network

- **Plain words:** `halve_timeseries.py` overwrote the base `.inp` when merely imported, which would change its hash and make the models refuse the network.
- **Changed:** explicit `--src/--dst/--force`, no side effects on import, refuses to overwrite (`57dc170`) (R02 §2).
- **What to say:** "A repository-integrity risk, not a result risk; closed."

### 4.7 M-07 / V-04 (🟠): model on disk differs from the promoted one; figures without provenance

- **Plain words:** in the audited checkout the joblib files did not match `promotion.json` (classifier `660be918…` vs declared `e2e7925e…`) (AUD §3.1).
- **In this worktree** the SHA-256 already matched; the discrepancy came from the outdated checkout. `verify_promotion("outputs/models")` returned `verified`, 0 discrepancies (R02 §3).
- **Changed:** `predict_network` checks the hashes before loading and stops with `ModelProvenanceError` on mismatch (warns if `promotion.json` is missing); figures get a `provenance.json` (`8f17147`, `0c8a3f4`); `promotion.json` is now versioned (`25eb169`, `e90448d`).
- **Not covered:** `ScenarioPredictor`, the SHAP analysis (SHapley Additive exPlanations, variable importance), `predict_from_inp` and the desktop map still load models without that check (R02 §3).
- **What to say:** "Prediction through the main path refuses a model that is not the promoted one; other loaders are a documented gap."

### 4.8 D-04 / H-06 (🟠/🟡): label resolution

Covered in §5. Status: investigated; path A recommended by the spike and chosen by the author on 2026-10-12 according to SPEC2, while R04 and deck slide 53 still call it pending (see §13.4).

### 4.9 Mistakes found in the audit itself (say this openly)

| Original statement | What was found | Source |
|---|---|---|
| The 72 cases "stops flooding when the factor rises" are "rounding noise" (D-04) | Only partially correct: 12 of the 72 were re-simulated; in all 12 the continuous volume really decreases, in 7 by ≥ 0.3 m³ (not a rounding effect), in 2 by < 0.1 m³ (borderline). The other 60 were not verified | R03 §4.1; AUD correction note after Table 4.1 |
| The FN in the 1–2 m³ band are "unstable by construction" | Exaggeration: the label is a deterministic function of the continuous volume | AUD §6.2 note |
| Read the volume from the binary `.out` for higher resolution | Wrong with the current 3-min report step: it departs from the continuous volume by up to 9.5 m³ per node | R03 §3.2 |
| "100 % of positives are integers" | 46,236 of 46,242 (consolidator's correction) | AUD §3.1 |
| Control sample "13 scenarios"; quantization bound "≤ 0.5 m³" | 17 simulated scenarios (15 with overflow); bound ≤ 0.51 m³ (node `8C`, 0.508 m³) | R04 §9 |
| Test baseline of 1,058 passing tests | 1,049: the count had included dots inside warning lines | R04 §7; R05 R4 |

**What to say:** "The audit was itself audited: when the spike contradicted two audit statements, we added dated correction notes rather than silently rewriting them."

### 4.10 Other audit findings worth knowing (not worked in this series)

- **H-01 (🟠):** all conduits start with `InitFlow` = sum of peaks at factor 1, in all runs: 855 m³ stored at t = 0, continuity error up to 12.4 %, +0.7 % to +10.5 % flooded volume in short events at f = 5 (auditor's figures, not re-simulated) (A01; R04 §6).
- **H-05 (🟡):** the 6 h "drain" adds a linear ramp to zero, +24.5 % input volume for `shape_2h` (A01).
- **H-08 (🟡):** the only capacity feature is the upstream one; the outlet conduit governs surcharge (A01).
- **D-02 / H-04 (🟠/🟡):** the continuity error is not stored in the DB (A03, A01).
- **M-05 (🟠):** `expm1(log1p)` retransformation bias and an underfitted regressor (A02).
- **M-15 (🟡):** non-nested Optuna tuning (A02) — relevant to §7.

**Test suite:** 881 tests at the audit (AUD §3.2); 1,049 passing at the start of the corrections; **1,174 passed, 0 failed** at the close of Part 1 (R05 header). An intermittent Tcl/Tk failure in `tests/desktop/test_results_tab.py` is environmental (R04 §7).

---

## 5. The label resolution spike (D-04 / H-06)

### 5.1 What the label really is

SWMM accumulates the overflow of each node continuously at every routing step. The `.rpt` prints *Total Flood Volume* in 10⁶ L with 3 decimals, i.e. **1 m³ resolution**. The pipeline reads it, multiplies by 1000 and labels `inunda = 1` if it is ≥ 1 m³ (R03 §2.1).

**Plain words:** the report rounds the continuous volume to the nearest whole m³, so "reported ≥ 1 m³" means "continuous ≥ about 0.5 m³".

### 5.2 Evidence (measured)

- In 15 re-simulated scenarios with per-node record (13 with overflow; 17 simulated in total), the label equals `V_continuous ≥ t` for any threshold t in **(0.4988, 0.5002] m³**, with **0 discrepancies in 2,400 node-scenario cases** (R03 §1, §4).
- In 1,407 of 1,408 informative cases (0 < V < 100 m³) the reported volume is the rounding of the continuous volume; 13 cases differ by 0.5–0.86 m³ (origin not verified), without affecting the label (R03 §1).
- The continuous volume is available from pyswmm (`node.statistics['flooding_volume']`) at ≤ 0.023 s per run (R03 §3.2).
- In the dataset: 1,894 rows have exactly 1.0 m³ and no row lies in (0, 1) m³ (R03 §2.2).
- Overflows labelled 0 are small: maximum 0.499 m³, median 0.153 m³ in the controls (R03 §4).

**Conclusion:** a deterministic quantization, **not noise**. The nominal threshold of 1 m³ works as ≈ 0.5 m³ continuous.

### 5.3 Effect on the evaluation (path A, measured from saved OOF)

XGBoost default, LeaveOneShapeOut (R03 §6.1):

| True reported volume | n | Missed (FN) | Found (TP) |
|---|---|---|---|
| 1 m³ (continuous ≈ 0.5–1.5) | 1,894 | 333 | 1,561 |
| [2, 5) m³ | 2,416 | 34 | 2,382 |
| [5, 100) m³ | 18,035 | 13 | 18,022 |
| ≥ 100 m³ | 23,880 | 0 | 23,880 |

Without the 1 m³ band, F1 goes from 0.9817 to 0.9847 and recall from 0.9918 to 0.9989; this is **only a sensitivity analysis**, because the exclusion is asymmetric: it removes the positive side of the boundary only (overflows of 0–0.5 m³ cannot be separated from dry nodes) (R03 §6.2).

### 5.4 Path A versus path B

| Path | What | Cost | Source |
|---|---|---|---|
| **A (recommended)** | keep the data; declare the effective ≈ 0.5 m³ threshold as an assumption; always report the confusion matrix with the band separated | no simulations, no code change mandatory | R03 §7 |
| B | re-simulate the 1,000 runs reading the continuous volume and choose a threshold before looking at metrics | about 17–20 min of SWMM (order of magnitude), but the dataset, DB, `prep_id`, benchmark, Optuna, validation and every figure must be redone | R03 §5 |

**Why A:** the label is already deterministic; with a continuous 1 m³ threshold the boundary just shifts (projected 2.3 %–3.6 % of positives would change; non-monotonic transitions fell from 9 to 5 in the re-simulated pairs, one of them new); and the regressor error at small volumes (median 1.6 m³, bias +2.6 m³ in the 1–5 m³ range) is far larger than the 0.5 m³ quantization (R03 §7). B becomes sensible only if the dataset is regenerated for another reason (e.g. H-01 InitFlow) (R03 §7.5).

**What to say:** "The label is a rounding of SWMM's own volume, so it is deterministic; we state the effective threshold of about half a cubic metre and report the borderline band separately. Re-simulating would cost minutes of SWMM but weeks of re-tracing every result, for the same phenomenon at a shifted threshold."

---

## 6. The approximate SVM

### 6.1 What an SVM is, for a hydraulics reader

- **Margin.** Picture the flooded and dry rows as points in a 21-dimensional space. A linear SVM draws the separating surface that keeps the widest "freeboard" (margin) to the nearest points of both classes.
- **Support vectors.** Only the points that touch or violate that freeboard define the surface; the others do not matter. That is where the name comes from.
- **Kernel trick.** When the boundary is curved, an SVM compares every scenario with every other through a similarity function. The **RBF kernel** is a bell-shaped similarity: two rows are similar if their features are close. `gamma` sets how narrow the bell is.
- **Hinge loss** is the penalty a margin classifier uses; **ε-insensitive loss** is the SVR (support vector regression) analogue: errors smaller than ε are ignored, larger ones grow linearly.

### 6.2 Why the exact SVM was replaced

- Exact `SVC`/`SVR` with RBF kernel scale as O(n²) to O(n³). The 2026-10-05 run accumulated **more than 1 day and 15 h of CPU (over 39 h) without finishing** and was cancelled (SPEC2 §3.1; deck slide 42 notes).
- **Illustrative arithmetic (not from a source file):** a full kernel matrix for 154,080 rows has about 2.4 × 10¹⁰ entries, roughly 190 GB in double precision; and the comparison needs 40 folds × 20 trials.

### 6.3 Before and after (plain words)

| | Before (exact) | After (approximate) |
|---|---|---|
| Kernel | exact RBF over all row pairs | **Nystroem map**: choose 100–400 landmark rows; describe every row by its RBF similarity to those landmarks |
| Classifier | `SVC(probability=True)` | linear SVM with **hinge** loss (SGD solver) on the Nystroem features, with `class_weight`, plus **Platt calibration** (`CalibratedClassifierCV(method="sigmoid", cv=3)`) to obtain probabilities |
| Regressor | exact `SVR` | linear **ε-insensitive SVR** (SGD) on the same map, target in `log1p` |
| Cost | did not finish | linear in rows; the 20-trial study took 829 s (RA `duration_s`) |

Sources: `swmm_resilience/ml/bench/models/svm_family.py` docstring; SPEC2 §3.1. It is **not an exact-kernel SVM** and must be called "approximate SVM (Nystroem + linear SVM)". Note: before Part 2, the WIP version used logistic loss, i.e. it was logistic regression, not an SVM (SPEC2 §2).

### 6.4 Known bias: Platt neutralizes `class_weight`

`class_weight` shifts the margin toward the minority class, but the Platt sigmoid is fitted on **unweighted** internal folds and removes most of that effect from the final probability. So the 0.5 threshold of the SVM is close to the unweighted decision, while XGBoost and the MLP shift their effective threshold with `scale_pos_weight`. Consequence: **the recall and F1 of the SVM are not directly comparable; compare classifiers with AUC**, which does not depend on the threshold (`svm_family.py` docstring "Sesgo conocido"; R06 §4.2). A second caveat: the internal calibration folds are not shuffled and depend on row order (R06 §5.9).

### 6.5 How it performed (CP)

| | Default | Tuned |
|---|---|---|
| Pooled RMSE (m³) | 222.46 | 214.87 |
| AUC | 0.9719 | 0.9626 |
| F1 | 0.8625 | 0.8302 |
| Oracle NSE | 0.5476 | 0.5831 |

The tuned SVM is **42.00 m³ (+24.3 %) worse than the no-learning baseline** (172.86 m³), and its AUC is below the baseline's 0.9844 as well (R06 §4.1). Tuning helped only 3.4 %, and F1/AUC got worse because the objective was the end-to-end RMSE, not F1 (R06 §3.2).

### 6.6 Plausible reasons (hypotheses, none tested)

1. Low-dimensional kernel approximation: the chosen classifier uses only 100 components, while XGBoost needs trees of depth 7–10 (R06 §4.2).
2. Narrow search space and small budget: 3–4 hyperparameters; the classifier optimum sits at the lower edge of `n_components` (100) and almost at the edge of `gamma` (1.13e-3 vs minimum 1e-3) (R06 §3.3).
3. Platt calibration cancels the class weighting (§6.4) — this explains why F1 is not comparable, not why AUC is low.
4. A linear ε-insensitive regressor on very skewed volumes (mostly 0, some thousands of m³) does not penalise large errors quadratically, unlike RMSE.

**What to say:** "The exact SVM is infeasible at this size, so we used a declared approximation. It is third and below the baseline. We have four plausible reasons, but we did not isolate any of them, so I present them as hypotheses."

Careful: slides 47 (old) and 53 (new) say "Attributable: approximate kernel, limited search space, and Platt calibration". Say "plausible" instead; nothing was attributed experimentally (R06 §4.2 heading: "hypotheses not tested here").

---

## 7. The fair comparison: XGBoost, MLP and approximate SVM

### 7.1 Design (what "fair" means here)

| Element | Value | Source |
|---|---|---|
| Data | `tabular_v4_21`, `prep_id ca706e27654e0cf5`, 154,080 rows, 963 runs | TP |
| Protocol | LeaveOneShapeOut, 40 folds | TP; PF |
| Main metric | pooled end-to-end volume RMSE | R06 §2 |
| Budget | 20 Optuna trials per family (10 random + 10 guided by TPE, Tree-structured Parzen Estimator, seed 42) | RA `n_trials`; R06 §2 |
| Tuning objective | the same end-to-end RMSE with an inner GroupKFold(5) grouped by shape | `scripts/ejecutar_parte2_v4_21.py` docstring |
| Fixed settings | XGBoost `n_jobs = 1`; MLP 200 epochs, batch 1,024 | RA `fixed_params` |
| Variants | `default` (config parameters + fixed settings) and `tuned` (best trial) | R06 §2 |
| Presentation rule | XGBoost and MLP always; SVM as third candidate whatever its result (author, 2026-10-12) | R06 §4.1; SPEC2 |

Why `n_jobs = 1`: inside the per-fold processes XGBoost runs with one thread anyway; fixing it makes tuning and evaluation reproducible and consistent (R06 §2).

### 7.2 Main results (CP, LeaveOneShapeOut)

| Model | Variant | Pooled RMSE (m³) | Per-fold mean RMSE (m³) | F1 | Recall | Precision | AUC | Oracle NSE | Total vol. error (%) |
|---|---|---|---|---|---|---|---|---|---|
| Nearest shape | no learning | 172.86 | 98.14 | 0.9764 | 0.9808 | 0.9721 | 0.9844 | 0.7102 | −2.98 |
| XGBoost | default | 112.80 | 58.65 | 0.9815 | 0.9919 | 0.9714 | 0.9993 | 0.8766 | −8.69 |
| **XGBoost** | **tuned** | **69.64** | **40.19** | 0.9812 | 0.9918 | 0.9707 | 0.9993 | **0.9530** | −2.95 |
| MLP | default | 114.38 | 64.08 | 0.9783 | 0.9889 | 0.9679 | 0.9992 | 0.8778 | −5.79 |
| MLP | tuned | 101.12 | 50.13 | 0.9783 | 0.9898 | 0.9672 | 0.9994 | 0.9018 | −5.67 |
| Approx. SVM | default | 222.46 | 142.64 | 0.8625 | 0.8544 | 0.8707 | 0.9719 | 0.5476 | −15.99 |
| Approx. SVM | tuned | 214.87 | 144.16 | 0.8302 | 0.8037 | 0.8586 | 0.9626 | 0.5831 | −15.81 |

Source: CP columns `rmse_vol_agregado_m3`, `rmse_vol_media_por_fold_m3`, `f1_agregado`, `recall_agregado`, `precision_agregada`, `auc_roc_agregado`, `nse_regresor_oracle`, `error_vol_total_pct`. In all seven, `folds_f1_indefinido = 0`.

**Reading.** Tuned XGBoost is 103.22 m³ (−59.7 %) below the baseline; the tuned MLP is 71.75 m³ (−41.5 %) below it and 31.47 m³ (+45.2 %) above tuned XGBoost (R06 §4.1). As classifiers, XGBoost and MLP are practically indistinguishable (AUC 0.9993 vs 0.9994); the difference is in the volume (oracle NSE 0.9530 vs 0.9018). The order XGBoost < MLP < baseline < SVM holds for both RMSE definitions.

**Do not mix:** XGBoost default here is 112.80 m³ (with `n_jobs = 1`) while Part 1 reported 106.21 m³ for XGBoost with `config.yaml` defaults and default `n_jobs` (CP1). Both are correct for their configuration (R06 §5.4).

### 7.3 Effect of tuning (derived from CP; R06 §3.2)

| Family | Pooled RMSE | Change | Oracle NSE | F1 |
|---|---|---|---|---|
| XGBoost | 112.80 → 69.64 | −38.3 % | 0.8766 → 0.9530 | 0.9815 → 0.9812 |
| MLP | 114.38 → 101.12 | −11.6 % | 0.8778 → 0.9018 | 0.9783 → 0.9783 |
| Approx. SVM | 222.46 → 214.87 | −3.4 % | 0.5476 → 0.5831 | 0.8625 → 0.8302 |

Tuning improves mainly the **regressor**; it barely changes the classifiers of XGBoost and MLP.

Best parameters (RA): XGBoost classifier depth 7, learning rate 0.0228, 239 trees, subsample 0.686; regressor depth 10 (upper edge of the space), learning rate 0.121, 346 trees, subsample 0.951; best inner RMSE 65.70 m³. MLP classifier layers (128, 64), regressor (64, 32, 16); best inner RMSE 77.20 m³. SVM best inner RMSE 202.40 m³. Convergence (CA): best trial number 10 (XGBoost), 11 (MLP), 17 (SVM); in XGBoost the first 5 random trials gave 311–331 m³ and 12 of the next 15 fell below 120 m³. Tuning time: 756 s, 3,101 s, 829 s (RA `duration_s`); not comparable across families because they ran simultaneously with different process counts (R06 §3.4).

### 7.4 Bootstrap intervals (R06 §5.3)

The 40 shapes were resampled with replacement 2,000 times (seed 42), pairing both models, and the pooled RMSE recomputed each time (disposable script, not versioned; reproduces 69.642 and 101.117).

| Quantity | Point (m³) | 95 % interval (m³) |
|---|---|---|
| Tuned XGBoost | 69.64 | 35.3 – 106.5 |
| Tuned MLP | 101.12 | 40.8 – 161.6 |
| Difference MLP − XGBoost | 31.47 | 4.4 – 56.2 |

Only 3 of 2,000 resamples gave a difference ≤ 0. **Intuition:** each model's interval is wide because it depends on whether `shape_quasi_uniform_block` is drawn; the paired difference stays positive because both models suffer on the same shapes. It covers shape-sampling variability only, not seed variability or tuning optimism, and the 40 shapes are not a random sample: an interpretation aid, not a formal test.

### 7.5 Where the error is (per shape; R06 §4.3, derived from PF)

| Model | Worst shape | Share of squared error | Its volume error | Top 3 share | Pooled RMSE without worst (m³) | Per-shape error range |
|---|---|---|---|---|---|---|
| XGBoost tuned | `shape_quasi_uniform_block` | 61.0 % | −31.2 % | 75.4 % | 44.06 | −31.2 % to +33.0 % |
| MLP tuned | `shape_quasi_uniform_block` | 73.0 % | −26.3 % | 84.2 % | 53.28 | −26.3 % to +61.6 % |

- Same three worst shapes in both tuned models: `shape_quasi_uniform_block`, `shape_twin_plateau_15h`, `shape_sustained_plateau_14h` — all sustained or plateau inflow.
- Tuning lowered XGBoost's RMSE on the worst shape from 600.6 to 337.6 m³ (PF `rmse_vol`).
- Without the worst shape, the XGBoost–MLP gap shrinks from 31.5 to 9.2 m³: much of XGBoost's advantage is decided on that one shape.
- Figure `docs/figures_part2/error_by_shape.png` (FIG §3): median absolute per-shape error 13.7 % (XGBoost), 10.5 % (MLP), 20.9 % (SVM); shapes with |error| > 20 %: 7 / 5 / 21. `shape_quasi_uniform_block` carries the largest SWMM volume (1.77 million m³); `common_bogota` is over-predicted by all three (+33.0 % / +61.6 % / +93.6 %) but is small (21,577 m³).

**Interpretation:** the shared weakness is extrapolating volume to unseen shapes with prolonged inflow; with only 40 shapes, holding out the one block-like shape removes much of the high-volume training space (FIG overall message). Not tested experimentally.

### 7.6 Parity plots and scenario totals (FIG §1–2)

- `docs/figures_part2/parity_nodes_3models.png`: node-level parity, log10(1 + volume) axes. R² 0.961 / 0.918 / 0.628; MAE 15.9 / 16.6 / 51.3 m³ (XGBoost / MLP / SVM).
- `docs/figures_part2/parity_scenario_totals_3models.png`: one marker per scenario (963). Median absolute percentage error over the 790 scenarios with SWMM total ≥ 1 m³: **XGBoost 15.5 %, MLP 12.7 %, SVM 30.4 %** (90th percentile 59.9 % / 51.5 % / 156.6 %). The SVM saturates at the top (five largest scenarios, 142,000–183,000 m³ in SWMM, predicted about 67,000–75,000 m³) and predicts volume in 102 of the 173 dry scenarios (XGBoost 3, MLP 2).

**Honest note:** the MLP has the lower **median scenario** error, while XGBoost has the lower **pooled node** RMSE. They answer different questions: "how well do I get the typical event total?" versus "how large are my node errors, weighted by size?". Do not claim XGBoost wins on every metric.

**Small count difference to know (378 vs 379).** Two ways of counting exist:

- **Classifier-label based** (primary; MC, BL, slides 40 and 46): a row counts as "predicted flooded" if the classifier says 1. Tuned XGBoost: FN 378, FP 1,383.
- **Routed-volume based** (FIG, slide 47): a row counts as "predicted flooded" if the end-to-end (routed) volume is > 0. Tuned XGBoost: 379 missed, 1,382 wrongly flooded.

They differ because in 2 rows the classifier says "flood" but the regressor returns a volume of 0, so the routed volume is 0: one truly flooded row moves from TP to "missed" and one dry row moves from FP to "correctly dry" (checked for this guide from `outputs/bench/parte2/ca706e27654e0cf5/xgboost/tuned/oof.parquet`). For the MLP (473 / 1,553) and the SVM (9,076 / 6,117) both counts coincide. Quote the classifier-based numbers as primary; if someone notices 379 on slide 47, give this explanation.

### 7.7 Flood maps (FIG §4; scenarios chosen by a rule, not by hand)

Rule: rank the 40 shapes by the total squared error of tuned XGBoost; show factor 3.0 (exists for every shape).

| Map file (in `docs/figures_part2/`) | Rule | SWMM total / flooded nodes | XGBoost | MLP | Approx. SVM |
|---|---|---|---|---|---|
| `flood_map_max_error_shape_quasi_uniform_block_f3.0.png` | largest error | 82,128 m³ / 61 | 59,408 (−27.7 %); +0 false, −0 missed; node RMSE 296.2 | 51,643 (−37.1 %); +4, −0; 473.3 | 59,610 (−27.4 %); +13, −7; 583.2 |
| `flood_map_median_error_shape_delayed_abrupt_peak_f3.0.png` | median (21 of 40) | 12,718 m³ / 60 | 14,108 (+10.9 %); +0, −0; 20.1 | 12,857 (+1.1 %); +0, −0; 21.4 | 11,576 (−9.0 %); +5, −10; 89.4 |
| `flood_map_min_error_shape_ultra_short_impulse_f3.0.png` | smallest, ≥ 20 flooded rows | 283 m³ / 36 | 423 (+49.4 %); +20, −1; 2.2 | 533 (+88.3 %); +23, −1; 3.7 | 1,499 (+429.6 %); +22, −4; 16.3 |

What to see: in the median case XGBoost and MLP reproduce the spatial pattern node by node; in the worst shape the pattern is right but magnitudes are low for all three (largest node about 6,000 m³ in SWMM, about 4,500 in XGBoost); the "best" case is best only in absolute m³ because volumes are tiny, and all models over-predict and raise 20–23 false alarms. Node profiles with the same scenarios: `node_profile_*.png` (not in the deck). Limits: one scenario per map, one factor, square-root colour scale.

### 7.8 Confusion matrices and the 1 m³ borderline band (MC, BL)

| Model | TN | FP | FN | TP | FN in 1 m³ band | FN outside band | FN ≥ 100 m³ |
|---|---|---|---|---|---|---|---|
| XGBoost tuned | 106,472 | 1,383 | 378 | 45,847 | 333 | 45 | 0 |
| MLP tuned | 106,302 | 1,553 | 473 | 45,752 | 354 | 119 | 4 |
| Approx. SVM tuned | 101,738 | 6,117 | 9,076 | 37,149 | 1,217 | 7,859 | 1,443 |
| Nearest shape | 106,556 | 1,299 | 889 | 45,336 | 611 | 278 | 0 |

**Reading:** 88 % of XGBoost's missed floods (333 of 378) are nodes at the minimum volume the label counts; none above 100 m³. The MLP misses 4 nodes above 100 m³. The SVM misses 9,076 flooded rows, 1,443 of them above 100 m³. False alarms outnumber missed floods for XGBoost and MLP (about 3.7 to 1 and 3.3 to 1; derived). **Interpretation:** for flood warning, missing a large flood is the costly error; XGBoost has none of those.

**What to say for §7:** "Same data, same protocol, same budget: tuned XGBoost is first on pooled volume RMSE and has no missed flood above 100 m³; the MLP is a close classifier and slightly better on the median scenario total; the approximate SVM is below the no-learning baseline. All three share the same weak shape, and these are exploratory estimates."

---

## 8. Limits and what NOT to claim

| Do not say | Say instead | Why (source) |
|---|---|---|
| "The model generalizes" | "On this network, the model predicts unseen hydrograph shapes better than a no-learning baseline" | one network, 160 nodes; no protocol holds out nodes or networks (R01 §0) |
| "XGBoost has 69.64 m³ error on new shapes" | "In an exploratory, conditionally tuned estimate, tuned XGBoost had a pooled RMSE of 69.64 m³" | hyperparameters chosen with all shapes (RA `status`; R06 §5.1) |
| "XGBoost is the best on every metric" | "XGBoost is best on pooled RMSE and oracle NSE; the MLP has the lower median scenario error" | FIG §2 |
| "LOSO F1 is 0.74" or "LOSO shows generalization" | "LOSO measures interpolation; pooled F1 there is 0.9667, and a lookup table does better" | CP1; R01 §3 |
| "The SVM is bad because of Platt / the kernel" | "Plausible reasons are…; none was isolated" | R06 §4.2 |
| "The SVM has lower recall than XGBoost" (as a fair comparison) | "Compare the SVM by AUC: 0.9626 vs 0.9993" | Platt bias (§6.4) |
| "The label is noisy" | "The label is a deterministic 1 m³ quantization, effective threshold ≈ 0.5 m³ continuous" | R03 §1 |
| "The tributary features improved NSE from 0.8141 to 0.8906" | "Between the two project states NSE went from 0.8141 to 0.8906; rows and contract changed together. The controlled comparison is per-shape mean RMSE 78.04 → 56.81 m³" | R04 §2 |
| "The intervals prove XGBoost is significantly better" | "A paired shape bootstrap places the MLP − XGBoost gap between 4.4 and 56.2 m³; an interpretation aid, not a formal test" | R06 §5.3 |
| "The model is valid for factors above 5" | "The training range is 0.2–5.0; no extrapolation protocol was run" | R01 §1 limits; INF0927 §8 |
| "Our resilience curves validate the model" | "Those curves are in-sample fit" | R02 §1 |
| "SVM durations show it is cheaper" | "Durations are not comparable across families" | R06 §3.4 |
| "This is the SVM" | "This is an approximate SVM (Nystroem + linear SVM)" | SPEC2 §3.1 |

Other limits to state if asked: one run and one seed; 20 trials; search optima at the edge of the space (XGBoost regressor depth, SVM `n_components`); 37 runs excluded and concentrated in short or extreme shapes, which are also the worst classified (R04 §5); InitFlow fixed at factor 1 (H-01, not re-simulated); MCC/PR-AUC not computed; implicit 0.5 decision threshold; older figures not regenerated.

---

## 9. Q&A bank

**Q1. Why did LOSO give a misleading F1?** Two separate issues. First, F1 was averaged per fold with undefined folds set to 0: 5 of 25 factors have no flooded node, so even a perfect model scored 0.80; the pooled F1 is 0.9667 (CP1; R01 §3). Second, LOSO measures interpolation: a lookup table gets 0.9718.

**Q2. What does LeaveOneShapeOut protect against?** Against the model seeing the same hydrograph shape at neighbouring factors. Each fold removes all 25 factors of one shape, so the model must use what it learned from the other 39 shapes. It does not protect against unseen networks, nodes or factors outside 0.2–5.0.

**Q3. Did you tune on the test shapes?** Partly, yes, and it is declared: Optuna used an inner GroupKFold(5) by shape over all 40 shapes, so the later LeaveOneShapeOut evaluation is not independent. The optimism applies equally to the three families, so the comparison is fair, but the absolute numbers are optimistic. A nested tuning (one study per held-out shape, ×40) is future work (R06 §5.1).

**Q4. Why is the SVM worse than the no-learning baseline?** Measured: pooled RMSE 214.87 vs 172.86 m³ and AUC 0.9626 vs 0.9844. Plausible, untested reasons: few Nystroem components (100 in the chosen classifier), a narrow search space with optima at its edge, a linear ε-insensitive regressor on very skewed volumes, and Platt calibration cancelling class weights (§6.6).

**Q5. Why not run the exact SVM?** Cost O(n²)–O(n³); the attempt ran more than 39 CPU-hours without finishing, and the comparison needs 40 folds × 20 trials (SPEC2 §3.1).

**Q6. Why choose XGBoost over the MLP when the MLP has the lower median scenario error?** The main metric, fixed before the run, is pooled node RMSE, where XGBoost is 31.47 m³ better with a paired bootstrap interval of 4.4–56.2 m³; it also has higher oracle NSE (0.9530 vs 0.9018) and no missed flood above 100 m³ (MLP: 4). The MLP's median scenario error is 12.7 % vs 15.5 %, so both remain candidates and both are always presented.

**Q7. What is the 0.5 m³ threshold?** The label is "reported volume ≥ 1 m³", but the report rounds to whole m³, so it equals "continuous volume ≥ t" with t in (0.4988, 0.5002] m³, with 0 discrepancies in 2,400 checked node-scenario cases (R03 §1).

**Q8. Why not re-simulate with the continuous volume?** SWMM would take about 17–20 min, but every downstream artefact (dataset, DB, `prep_id`, benchmark, Optuna, validation, all figures) would have to be redone, and the boundary problem would only move to another threshold (R03 §5, §7).

**Q9. What does the confusion matrix say about missed floods versus false alarms?** Tuned XGBoost: 378 missed, 1,383 false alarms. 333 of the missed are at the minimum labelled volume (1 m³) and none exceeds 100 m³. Recall 0.9918, precision 0.9707 (MC; BL; CP).

**Q10. Is the model valid for another network?** Not shown. All results are for one network; features such as tributary volume are transferable in definition, but no test holds out a network (R06 §5.8).

**Q11. What about extrapolation beyond factor 5?** Not measured. Training covers 0.2–5.0; the code only warns outside that range; no extrapolation protocol was run (R01 §1 limits).

**Q12. Why is the error concentrated in one shape?** Measured: `shape_quasi_uniform_block` carries 61.0 % of tuned XGBoost's squared error and is underestimated by 31.2 %; it has the largest SWMM volume (1.77 million m³). Interpretation: holding out the only block-like, high-volume shape removes much of the high-volume training space.

**Q13. Is the difference between XGBoost and MLP robust?** Under shape resampling, yes as an interpretation aid (3 of 2,000 resamples ≤ 0). It ignores seed variability and tuning optimism.

**Q14. What is "oracle NSE"?** The regressor's NSE on the rows that really flooded, as if the classifier were perfect. It isolates volume skill.

**Q15. Why two RMSE numbers (pooled vs per-fold mean)?** Pooled weights shapes by their error (main metric); per-fold mean weights shapes equally. Ranking is the same with both (R06 §5.7).

**Q16. Why does XGBoost default differ between reports (106.21 vs 112.80 m³)?** Different configuration: Part 2 fixed `n_jobs = 1` for reproducibility; Part 1 used config defaults (R06 §5.4).

**Q17. What did you change in inference earlier (shape features)?** On 2026-09-27, prediction stopped setting duration and time-to-peak to 0 and computes them from the `.inp`. In the current `tabular_v4_21` contract those two inputs were replaced by the 6 tributary features (INF0927 §4; `contracts.py`).

**Q18. What is the computational cost?** Tuning (20 trials): 756 s XGBoost, 3,101 s MLP, 829 s SVM on a 64-CPU machine, run simultaneously, so not comparable (RA; R06 §3.4). Re-simulating 1,000 SWMM runs: about 17 min (R03 §5.1).

**Q19. Is it reproducible?** Seed 42; run commit `8a06605` with `git_dirty: false`; SHA-256 of every OOF and table in TP; versions Python 3.12.10, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.6.0, optuna 5.0.0 (R06 §7). One seed only.

**Q20. Are the bench predictions in git?** No, `outputs/` is ignored; they are regenerated with the commands in §13 and checked against the recorded SHA-256 (R04 §5.10).

**Q21. Which model is used for the maps in the app today?** The promoted XGBoost `tabular_v4_21` of 2026-10-06, verified against `promotion.json`; the Part 2 tuned models were not promoted. Some loaders (desktop map, SHAP, `ScenarioPredictor`) do not run the verification (R02 §3).

**Q22. Did the audit find problems in the SWMM data?** No invalidating defect: re-simulations reproduce the stored labels to the m³. Hydraulic assumptions to declare: InitFlow fixed at factor 1 (H-01), the 6 h ramp (H-05), upstream capacity only (H-08), 1 m³ label resolution (AUD §9).

**Q23. Why 963 and not 1,000 runs?** 37 runs exceed 5 % continuity error, mostly at factor 0.2 and short or extreme shapes (worst 12.38 %, `shape_ultra_short_impulse` at 0.2; deck slide 32).

**Q24. What remains?** Confirm label path A in writing; nested tuning; more seeds; wider search spaces (SVM); the 7 reserved validation shapes; a factor-extrapolation test; H-01 sensitivity; MCC/PR-AUC and a threshold chosen by inner CV; regenerate older figures with the declared model; push the branch.

**Q25. Why should we believe the audit corrections?** Each task was implemented, reviewed for spec compliance and quality, corrected and re-reviewed; numbers were recomputed independently by reviewers; tests went from 1,049 to 1,174 passing (R05).

---

## 10. Slide-by-slide walkthrough

Deck: `Presentacion_MLSMFP_20-10_avances_auditoria_svm.pptx`, 53 slides (read from a copy; the progress slides are 38–53). Slides 38–46 are identical in content to slides 38–46 of the 13-10 deck except the footer date ("October 2026"); 47–52 are new; 53 replaces the old conclusions.

### 10.1 Progress slides (38–53)

**Slide 38 — Audit: what was found and what was fixed.** Message: seven findings, five fixed or partial, two open. Say: "The October 10 audit produced 111 findings; these seven affect how results are read. M-01 is the key one: in LOSO a baseline that only interpolates reaches 9.13 m³ against 57.89 m³ for XGBoost. M-03 was an aggregation artefact: F1 0.7379 becomes 0.9667 pooled. M-02 remains open, and the label decision follows the spike." Numbers: 9.13 / 57.89 (CP1), 71.6 % (R04 §2), 0.7379 → 0.9667 (CP1). Likely question: "Why does the slide say D-04 pending?" (see §13.4).

**Slide 39 — What each protocol measures.** Message: LOSO/GroupKFold5 = interpolation, LeaveOneShapeOut = unseen shapes. Say: "In LOSO the baseline wins, so LOSO proves nothing about learning. In LeaveOneShapeOut the order reverses: 172.86 for the nearest-shape baseline against 106.21 for default XGBoost." Numbers: CP1. Likely question: "Is GroupKFold5 harder?" — No, identical interpolation predictions.

**Slide 40 — XGBoost confusion matrix on unseen shapes (default).** Say: "45,845 TP, 106,526 TN, 1,329 FP, 380 FN; recall 0.9918, precision 0.9718. 333 of the 380 misses are nodes at 1 m³, none above 100 m³." Numbers: MC1, BL1. Note: this is default XGBoost (Part 1); slide 46 shows the tuned models. Likely question: "What is the F1 without the band?" — 0.9847, sensitivity only.

**Slide 41 — The flood label and its resolution.** Say: "SWMM computes a continuous volume, the report prints whole m³, and the label is ≥ 1 m³; that equals about 0.5 m³ continuous, with no discrepancy in 2,400 checked cases. Path A states the assumption; path B re-simulates and redoes everything." Numbers: R03 §1, §5. Likely question: Q7/Q8.

**Slide 42 — Approximate SVM: architecture change.** Say: "The exact RBF SVM did not finish after more than 39 CPU-hours. We approximate the kernel with a Nystroem map and train a linear hinge SVM with Platt calibration, and a linear ε-insensitive SVR. It is an approximate SVM, not an exact-kernel SVM." Numbers: SPEC2 §3.1; "~8–13 s per fold" on the slide is from the cost spike and was not re-checked for this guide; study ≈ 14 min = 829 s (RA). Likely question: Q5.

**Slide 43 — How the fair comparison is run.** Say: "Same data, same 20-trial budget with the same inner CV by shape, same protocol, same metrics. Hyperparameters were chosen with all shapes, which is optimistic equally for the three." Numbers: TP, RA. Likely question: Q3.

**Slide 44 — Results on unseen shapes.** Say: "Tuned XGBoost 69.64 m³, 38 % below its default; tuned MLP 101.12; both F1 about 0.98, AUC about 0.999. The approximate SVM is below the baseline: 214.87 against 172.86." Numbers: CP. Likely question: Q16 (default XGBoost 112.80 vs 106.21).

**Slide 45 — Optuna tuning: 20 trials per family.** Say: "Best inner RMSE: 65.7 XGBoost after 12.6 min, 77.2 MLP after 51.7 min, 202.4 SVM after 13.8 min. The inner value is GroupKFold by shape, not the LeaveOneShapeOut figure." Numbers: RA `best_value`, `duration_s`; CA. Likely question: "Would more trials change the order?" — possibly for SVM/MLP (R06 §5.2).

**Slide 46 — Confusion matrices: tuned XGBoost and MLP.** Say: "They classify almost the same. XGBoost: 1,383 false alarms, 378 misses, 333 at 1 m³, none above 100 m³. MLP: 1,553 and 473, 354 at 1 m³, 4 above 100 m³." Numbers: MC, BL. Likely question: Q9.

**Slide 47 — Parity per node (new).** Say: "Every node-scenario row, out-of-fold. XGBoost and MLP follow the diagonal over four orders of magnitude; R² 0.961, 0.918, 0.628. The SVM misses 9,076 flooded rows. RMSE is dominated by few large rows, so read it with the next slide." Numbers: FIG §1. Caution: the slide says 379 missed for XGBoost; the matrix says 378 (§7.6). Likely question: "Why log axes?" — to show zeros and 10,000 m³ together; small errors look larger.

**Slide 48 — Parity per scenario (new).** Say: "One marker per scenario. Median error 15.5 % XGBoost, 12.7 % MLP, 30.4 % SVM. Honestly, the MLP is better on the median scenario, XGBoost on pooled RMSE: different questions. The SVM saturates on the biggest scenarios and predicts volume in 102 of 173 dry scenarios." Numbers: FIG §2. Likely question: Q6.

**Slide 49 — Error by shape (new).** Say: "Failures concentrate in a few shapes. The quasi-uniform block is under-predicted by all three and carries 1.77 million m³, so it dominates the pooled RMSE. `common_bogota` is over-predicted but small." Numbers: FIG §3. Likely question: Q12.

**Slide 50 — Flood map, largest-error shape (new).** Say: "Chosen by rule, not by hand. XGBoost finds exactly the 61 flooded nodes, no false, no missed, but underestimates the total by 27.7 %; all three underestimate." Numbers: FIG §4. Likely question: "Is this the worst factor?" — factor 3.0 was fixed for all maps.

**Slide 51 — Flood map, median-error shape (new).** Say: "For a typical shape XGBoost and MLP reproduce the map node by node, node RMSE about 20 m³; the SVM misses 10 nodes and invents 5." Numbers: FIG §4.

**Slide 52 — Flood map, smallest-error shape (new).** Say: "Smallest squared error only because volumes are tiny, 283 m³ in total. In relative terms every model over-predicts and raises 20–23 false alarms: short impulses are hard." Numbers: FIG §4. Likely question: "Why not show a really good case?" — the rule is documented; this shape is also among the worst classified (PF1 F1 0.8798 in Part 1).

**Slide 53 — Conclusions and next steps.** Say: "Tuned XGBoost is best on pooled RMSE and node parity; the MLP has the better median scenario error; the approximate SVM is below the baseline. Scope: exploratory, one network, 963 runs. Next: confirm the label path, nested tuning, wider SVM search, confusion matrix for extreme shapes, per-family maps, regenerate the two Spanish pictures." Caution: replace "Attributable" with "plausible reasons" when speaking (§6.6), and say that path A was already chosen on 2026-10-12 if you confirm that decision (§13.4). Likely question: Q24.

### 10.2 Base slides 1–37 (earlier material)

| Slide | Topic | Spanish text in pictures? |
|---|---|---|
| 1 | Title (dated 2026-09-15) | — |
| 2 | Thesis II schedule (Gantt) | English picture now (`slide02_gantt_schedule_en.png`) |
| 3–4 | Hydrograph shape panels (8 existing + 10 new) | English now (`slide03_hydrograph_shapes_en.png`) |
| 5 | New hydrograph shapes (8), 2026-09-01 | **Yes**: picture title in Spanish; not regenerated (shapes not in versioned data) |
| 6 | Newest 10 training shapes | English now (`slide06_new_training_shapes_en.png`) |
| 7–10 | Hydrograph 1 and 2: maps and parity | No |
| 11 | 7 reserved validation hydrographs (5 interpolation, 2 extrapolation) | English now (`slide11_validation_shapes_en.png`) |
| 12–14 | Committee observations 1–21 | — |
| 15 | Training and validation workflows | English now (two flowcharts) |
| 16 | Example database, node 75 | No |
| 17 | Is it worth replacing XGBoost? (routes and effort) | — |
| 18 | What we have and what remains to be measured | — |
| 19–23 | Literature P1–P5 (Zhang 2024, Palmitessa 2022, Garzón 2024 and 2026, Brazão 2026) | — |
| 24–25 | Data, resources and effort per paper | — |
| 26 | Criteria for investing more time | — |
| 27 | Computational budget | — |
| 28–29 | References | — |
| 30 | Single-scenario result `long_moderate` f 2.0 (SWMM 12,738 vs XGBoost 11,103 m³) | No. Caution: a training scenario, so probably an in-sample fit (V-01); not re-checked |
| 31 | First real end-to-end run (1,000 runs, 0 failures) | — |
| 32 | Continuity exclusion: 37 of 1,000 runs | — |
| 33 | 5-family benchmark (XGBoost 78.04, MLP 99.61, RF 185.46, Linear 201.18; SVM pending) | — . Superseded by slides 44–46. Interpretation: 78.04 matches the per-shape mean RMSE of `tabular_v3_17` (78.0368), so these look like per-fold means, not comparable with the pooled 69.64 |
| 34 | Parity by family (OOF) | **Yes**: titles contain "todas_las_corridas"; not regenerated (would need full re-evaluation) |
| 35 | Earlier Optuna study on XGBoost (50 trials, inner RMSE 108.84) | — . Earlier study, not the Part 2 one |
| 36 | Tributary features worth it (81.95 → 59.07, 150,240 rows, exploratory protocol) | — . The official controlled figure is 78.0368 → 56.8088 on LeaveOneShapeOut |
| 37 | Next steps of 2026-10-05 | — . Items 1–2 done since |

Sources: deck text read with python-pptx; picture fingerprints compared with `docs/figures_en/*.png` (slides 2, 3, 4, 6, 11, 15 match the English files exactly); `docs/figures_en/INDEX.md` for slides 5 and 34.

---

## 11. Numbers cheat sheet

| # | Value | Meaning | Source (column) |
|---|---|---|---|
| 1 | 160 | junction nodes of Chicó Sur | A01 header |
| 2 | 40 × 25 = 1,000 | shapes × load factors = SWMM runs | A01; TP `n_runs_full_grid` |
| 3 | 0.2–5.0, step 0.2 | factor range | FLUJO_ACTUAL §1 |
| 4 | 37 / 963 | runs excluded (continuity > 5 %) / kept | TP `n_missing_cells`, `n_runs` |
| 5 | 154,080 | node-scenario rows | TP `n_rows` |
| 6 | 46,225 | flooded rows | R03 §2.2; MC |
| 7 | 21 | features in `tabular_v4_21` | provenance `n_features` |
| 8 | 111 (2 🔴, 25 🟠) | audit findings | AUD §4 |
| 9 | 9.13 vs 57.89 m³ | LOSO: interpolation baseline vs default XGBoost | CP1 `rmse_vol_agregado_m3` |
| 10 | 0.9718 vs 0.9667 | LOSO pooled F1: baseline vs XGBoost | CP1 `f1_agregado` |
| 11 | 0.7379 → 0.9667 | LOSO F1 zero-filled vs pooled (XGBoost) | CP1 |
| 12 | 0.8000 | perfect predictor's zero-filled LOSO F1 | R01 Table 3.1 |
| 13 | 172.86 m³ | nearest-shape baseline, pooled RMSE | CP |
| 14 | 106.21 m³ | default XGBoost (Part 1) LeaveOneShapeOut | CP1 |
| 15 | 78.0368 → 56.8088 m³ | controlled contract comparison (per-shape mean) | explicacion-completa §11.1 |
| 16 | 0.8141 → 0.8906 | oracle NSE, audit → Part 1 (uncontrolled) | R04 §2 |
| 17 | 69.64 m³ | tuned XGBoost pooled RMSE | CP |
| 18 | 101.12 m³ | tuned MLP pooled RMSE | CP |
| 19 | 214.87 m³ | tuned approximate SVM pooled RMSE | CP |
| 20 | 112.80 / 114.38 / 222.46 m³ | default XGBoost / MLP / SVM (Part 2) | CP |
| 21 | −38.3 % / −11.6 % / −3.4 % | effect of tuning on pooled RMSE | derived from CP |
| 22 | 40.19 / 50.13 m³ | per-fold mean RMSE, tuned XGBoost / MLP | CP `rmse_vol_media_por_fold_m3` |
| 23 | 0.9812 / 0.9783 / 0.8302 | pooled F1 tuned XGBoost / MLP / SVM | CP `f1_agregado` |
| 24 | 0.9993 / 0.9994 / 0.9626 | AUC tuned XGBoost / MLP / SVM | CP `auc_roc_agregado` |
| 25 | 0.9844 | baseline "AUC" (not interpreted: no probabilities) | CP |
| 26 | 0.9530 / 0.9018 / 0.5831 | oracle NSE tuned | CP `nse_regresor_oracle` |
| 27 | −2.95 % / −5.67 % | total volume error tuned XGBoost / MLP | CP `error_vol_total_pct` |
| 28 | 106,472 / 1,383 / 378 / 45,847 | TN / FP / FN / TP tuned XGBoost | MC |
| 29 | 333 of 378; 0 ≥ 100 m³ | XGBoost misses at 1 m³; large misses | BL |
| 30 | 4 | MLP misses ≥ 100 m³ | BL `fn_con_vol_ge_100m3` |
| 31 | 9,076 / 1,443 | SVM misses / misses ≥ 100 m³ | MC; BL |
| 32 | 61.0 % / 73.0 % | worst-shape share of squared error, XGBoost / MLP | derived from PF |
| 33 | −31.2 % | tuned XGBoost volume error on `shape_quasi_uniform_block` | PF `error_vol_pct` |
| 34 | 44.06 m³ | tuned XGBoost RMSE without the worst shape | derived from PF |
| 35 | 31.47 m³ (4.4–56.2) | MLP − XGBoost gap and 95 % shape-bootstrap interval | R06 §5.3 |
| 36 | 15.5 % / 12.7 % / 30.4 % | median scenario error XGBoost / MLP / SVM (790 scenarios) | FIG `scenario_median_ape_pct` |
| 37 | 0.961 / 0.918 / 0.628 | node parity R² | FIG `pooled.r2` |
| 38 | 20 trials (10 + 10) | Optuna budget per family | RA `n_trials` |
| 39 | 65.70 / 77.20 / 202.40 m³ | best inner RMSE (GroupKFold 5 by shape) | RA `best_value` |
| 40 | (0.4988, 0.5002] m³ | effective label threshold (continuous volume) | R03 §4 |
| 41 | 2,400 / 0 | checked node-scenario cases / discrepancies | R03 §4 |
| 42 | 1,894 | rows with exactly 1 m³ | R03 §2.2 |
| 43 | 0.9817 → 0.9847 | F1 without the 1 m³ band (sensitivity, default XGBoost) | R03 §6.2 |
| 44 | ≈ 17–20 min | SWMM time to re-simulate 1,000 runs | R03 §5.1 |
| 45 | > 39 CPU-h | exact SVM attempt, not finished | SPEC2 §3.1 |
| 46 | 1,049 → 1,174 | passing tests, start → close of Part 1 | R05 |

---

## 12. Glossary English and Spanish, and speaking tips

| English | Spanish |
|---|---|
| surrogate model | modelo sustituto (surrogate) |
| flood volume | volumen de inundación |
| flood label | etiqueta de inundación |
| node / junction | nodo / pozo (unión) |
| outfall | descarga / punto de salida |
| conduit / pipe | conducto / tubería |
| hydrograph shape | forma de hidrograma |
| load factor (flow multiplier) | factor de carga (multiplicador de caudal) |
| peak flow | caudal pico |
| time to peak | tiempo al pico |
| tributary (upstream) flow | caudal tributario (aguas arriba) |
| upstream capacity | capacidad aguas arriba |
| continuity error | error de continuidad |
| dynamic wave routing | tránsito por onda dinámica |
| report step / routing step | paso de reporte / paso de tránsito |
| feature | variable de entrada (feature) |
| feature contract | contrato de variables |
| target | variable objetivo |
| classifier / regressor | clasificador / regresor |
| two-stage model | modelo en dos etapas |
| end-to-end prediction | predicción de extremo a extremo |
| cross-validation | validación cruzada |
| fold | pliegue (fold) |
| out-of-fold (OOF) prediction | predicción fuera de pliegue |
| held-out | retenido / apartado |
| no-learning baseline | referencia sin aprendizaje |
| nearest shape | forma más cercana |
| factor interpolation | interpolación entre factores |
| unseen shape | forma no vista |
| pooled metric | métrica agregada |
| per-fold mean | media por pliegue |
| zero-filled | rellenado con ceros |
| confusion matrix | matriz de confusión |
| true / false positive | verdadero / falso positivo |
| false alarm | falsa alarma |
| missed flood | inundación no detectada |
| recall / precision | sensibilidad (exhaustividad) / precisión |
| specificity | especificidad |
| borderline band | banda limítrofe |
| hyperparameter tuning | ajuste (afinación) de hiperparámetros |
| trial | ensayo (trial) |
| search space | espacio de búsqueda |
| nested tuning | ajuste anidado |
| conditional exploratory estimate | estimación exploratoria condicionada |
| support vector machine | máquina de vectores de soporte |
| margin / support vectors | margen / vectores de soporte |
| kernel trick | truco del kernel |
| Platt calibration | calibración de Platt |
| bootstrap resampling | remuestreo bootstrap |
| in-sample fit | ajuste en muestra |
| provenance / fingerprint (SHA-256) | procedencia / huella (SHA-256) |
| spike (bounded investigation) | investigación acotada (spike) |
| quantization | cuantización |
| decision log | bitácora de decisiones |
| parity plot | gráfico de paridad |
| node-scenario row | fila nodo-escenario |

**Speaking tips.**

- Say "**pooled**" (rhymes with "cooled"), never "pulled". "Out-of-fold" is three words, stress on "fold".
- "**Nystroem**" is pronounced roughly "NEE-strum". "**Optuna**" "op-TOO-na". "**XGBoost**" "ex-gee-boost". "**SWMM**" is said "swim".
- Numbers: "sixty-nine point six four cubic metres"; "zero point nine eight one two"; "one hundred fifty-four thousand and eighty rows". Decimal **point**, thousands **comma**.
- "**Leave-one-shape-out**" said as a phrase; avoid saying "LOSO" without "by factor" the first time.
- Useful hedges: "we measured…", "a plausible explanation, not tested, is…", "within this network…", "as an exploratory estimate…".

---

## 13. Where everything is, repository state and ready-to-present checklist

### 13.1 Files

| What | Path |
|---|---|
| Original audit and annexes | `docs/auditoria/2026-10-10-auditoria-completa.md`, `docs/auditoria/anexos/01-…05-….md` |
| Corrections (Part 1) | `docs/auditoria/correcciones/01-evaluacion-honesta.md` … `05-bitacora-de-decisiones.md` |
| Model selection (Part 2) | `docs/auditoria/correcciones/06-seleccion-de-modelos.md` |
| Data tables | `docs/auditoria/correcciones/datos/` (Part 1) and `datos_parte2/` (Part 2), each with provenance JSON |
| Figures Part 2 | `docs/figures_part2/` (`INDEX.md`, `figure_numbers.json`, 9 PNG) |
| English base-deck pictures | `docs/figures_en/` (`INDEX.md`, 6 PNG) |
| Specs | `docs/superpowers/specs/2026-10-11-correcciones-auditoria-design.md`, `2026-10-12-parte2-svm-comparacion-justa-design.md` |
| Approximate SVM | `swmm_resilience/ml/bench/models/svm_family.py` |
| Baselines | `swmm_resilience/ml/bench/baselines.py` |
| Feature contract | `swmm_resilience/ml/contracts.py`, `swmm_resilience/extraction/tributary_features.py` |
| Run scripts | `scripts/ejecutar_evaluacion_v4_21.py`, `scripts/informe_evaluacion.py`, `scripts/ejecutar_parte2_v4_21.py` |
| Figure scripts | `scripts/figures_part2/make_parity_and_maps.py`, `scripts/figures_en/*.py` (README in each folder) |
| OOF predictions (ignored by git) | `outputs/bench/oof/ca706e27654e0cf5/`, `outputs/bench/parte2/ca706e27654e0cf5/<family>/<stage>/oof.parquet` |

### 13.2 Commands to reproduce (from the worktree root, Python `../../../venv/Scripts/python.exe`)

```
# Part 1 (default XGBoost + baselines, 3 protocols; about 6 min) and its tables (about 1.5 min)
../../../venv/Scripts/python.exe scripts/ejecutar_evaluacion_v4_21.py
../../../venv/Scripts/python.exe scripts/informe_evaluacion.py --prep-id ca706e27654e0cf5 --prepared-dir outputs/bench/prepared_v4_21 --oof-root outputs/bench/oof --out docs/auditoria/correcciones/datos --flood-threshold 1.0

# Part 2 (one process per family; finished stages are skipped) and its tables
../../../venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --family xgboost --eval-jobs 16
../../../venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --family mlp
../../../venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --family svm --eval-jobs 16
../../../venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --build-tables

# Figures: see scripts/figures_part2/README.md and scripts/figures_en/README.md
```

Sources: `docs/auditoria/correcciones/datos/README.md`; R06 §7.

### 13.3 Repository state (2026-10-20, when this guide was written)

- Branch `worktree-optuna-tuning-familia-ganadora`, tracking `origin/fix/cierre-auditoria-hidrogramas-20260927`, **50 commits ahead, not pushed** (push requires the author's authorization; R05 R3).
- Last commits: `86cec4d` (figures Part 2), `beb4c19` (English figures), `2f51e18`, `5bd3e03`, `1875367`, `7b0b7f5` (English messages), `6c68984`, `b2dd4c9` (English documentation), `817309e` (English deck builder).
- Untracked: `.superpowers/`, `Presentacion_MLSMFP_13-10_avances_auditoria_svm.pptx` and the new `Presentacion_MLSMFP_20-10_avances_auditoria_svm.pptx`.
- Decision log: the versioned `05-bitacora-de-decisiones.md` contains rulings R1–R14; no R15–R18 were found in the versioned documents.

### 13.4 Pending items

| Item | State | Source |
|---|---|---|
| Label path A/B | SPEC2 records the author's decision "path A" on 2026-10-12, but R04, the memory notes and slide 53 still list it as pending. Confirm it explicitly with the advisor and update R04/slide 53 | SPEC2 header; R04 §4 |
| Push | 50 commits unpushed | git |
| Nested tuning by shape, more seeds | not done | R06 §5 |
| Wider search spaces (SVM, XGBoost regressor depth at the edge) | not tested | R06 §3.3 |
| 7 reserved validation shapes | no results table in the sources reviewed | §1.7 |
| Figures not regenerated with the declared model; slides 5 and 34 still with Spanish picture text | open | R02 §1, §3; `docs/figures_en/INDEX.md` |
| Translating older Spanish documents (e.g. README, FLUJO_ACTUAL, DOCUMENTACION_COMPLETA, resultados) | not done | file inspection |
| `que_mide` text of GroupKFold5 in Part 1 CSV, H-07 parser, MCC/PR-AUC, threshold by inner CV | open minors | R04 §6, §9 |

### 13.5 Ready-to-present checklist

- [ ] I can explain LOSO vs LeaveOneShapeOut with the 100/140 m³ toy example in under one minute.
- [ ] I can explain why a perfect model scored 0.80 in LOSO.
- [ ] I know the main table: 69.64 / 101.12 / 214.87 / 172.86 m³ and what "pooled" means.
- [ ] I will say "exploratory, conditional on tuning with all shapes" whenever I quote tuned numbers.
- [ ] I will say "MLP has the lower median scenario error" before the advisor finds it.
- [ ] I will say "plausible reasons" (not "attributable") for the SVM.
- [ ] I know the label story: rounding → ≈ 0.5 m³, deterministic, path A, band reported separately.
- [ ] I know the two numbers that look inconsistent and why: 106.21 vs 112.80 (n_jobs), 378 vs 379 (figure definition).
- [ ] I have decided how to present path A (decided on 2026-10-12 or still open) and will say it consistently.
- [ ] I can list what was not done: nested tuning, other networks, factor extrapolation, 7 validation shapes, H-01.
- [ ] I checked slides 5 and 34 (Spanish picture text) and slide 30 (probably in-sample) before showing them.
- [ ] The deck file opened is `Presentacion_MLSMFP_20-10_avances_auditoria_svm.pptx` (53 slides).
