# Full project audit — technical and hydraulic part

**Date:** 2026-10-10
**Branch:** `fix/cierre-auditoria-hidrogramas-20260927`, base commit `6e8579b`, **plus** the 16 uncommitted files of Phase A (`LeaveOneShapeOut`, inclusive flooding rule, promotion disabled).
**Mode:** read-only. No code, `config.yaml` or `outputs/` was modified and no commit was made. All new material is in `docs/auditoria/`.
**Deliverable:** this report (consolidated and verified) + 5 annexes with the evidence, the scripts and the detail of each finding.

| Annex | Area | Findings |
|---|---|---|
| [01-hidraulica-swmm](anexos/01-hidraulica-swmm.md) | Hydraulics / SWMM | H-01…H-14 |
| [02-ml-protocolo-evaluacion](anexos/02-ml-protocolo-evaluacion.md) | ML, evaluation protocol, confusion matrix | M-01…M-22 |
| [03-datos-persistencia-sql](anexos/03-datos-persistencia-sql.md) | Data and SQLite v17 | D-01…D-22 |
| [04-cierre-auditorias-previas-y-documentacion](anexos/04-cierre-auditorias-previas-y-documentacion.md) | Traceability of previous audits, orphans, documentation | T-01…T-19 + 117 traced items |
| [05-analisis-visualizacion-soporte](anexos/05-analisis-visualizacion-soporte.md) | Analysis, visualization, GUI, packaging | V-01…V-34 |

---

## 1. Executive summary

**Verdict.** The hydraulic data-generation chain is sound: the labels are not invalidated. Independent SWMM re-simulations reproduce the v17 DB totals to the m³ in 11 shape/factor combinations, the hydrograph rescaling preserves volume (error ≤ 0.08%) and at factor 1.0 (flow multiplier) the pipeline reproduces the original `.inp`. The data are intact (160,000 rows = 40 shapes × 25 factors × 160 nodes, no duplicates or orphans, `inunda` matches `vol ≥ 1` in every row).

**The problem is not in the data but in what is concluded from them.** There are two critical findings, both in the evaluation protocol, that change how the thesis results must be read:

1. **M-01 — The LOSO and GroupKFold5 protocols only measure interpolation.** They group by `factor_mult`, and with a dense grid of 25 factors a no-learning lookup table (interpolating between the neighboring factors of the same node and shape) beats XGBoost: volume RMSE **9.0 m³ versus 68.4 m³**, F1 **0.972 versus 0.966**. Those figures do not show that the model has learned hydraulics.
2. **M-02 — With unseen hydrograph shapes, volume generalizes poorly.** Regressor NSE **0.814** (versus 0.953 in LOSO) and total-volume error per shape from **−63% to +216%**. A single shape concentrates 61% of the squared error. The most likely cause (H-02/M-02) is that the shape enters the model only as `duracion_horas` and `tiempo_al_pico_h`, with no **event volume** variable.

What does generalize well is the flooding / no-flooding **classification** with new shapes (pooled F1 0.979, recall 0.992). The weakness is in the magnitude of the volume.

**Seven other major findings relevant to the thesis** (full table in section 4):

| ID | What | Why it matters |
|---|---|---|
| M-03 | F1/precision/recall are averaged per fold with `zero_division=0` | In LOSO, F1 = 0.74 is reported when the pooled value is 0.966. A perfect predictor would score 0.80. The LOSO vs GroupKFold5 "gap" is an artifact |
| M-07 / V-04 | The default pipeline overwrites `outputs/models` and `promotion.json` is left false | The model on disk (`660be918…`) is not the promoted one (`e2e7925e…`). Maps, SHAP and validations use a different model from the one cited |
| V-01 / M-13 | Resilience curves, total volume and `--evaluate-shapes` use training data without labeling it | They can be read as validation and are in-sample fit |
| H-01 | All conduits start with `InitFlow` = sum of peaks at factor 1 | 855 m³ stored at t=0; continuity error up to 12.4%; in short events at f=5 the flooded volume rises +0.7% to +10.5% |
| H-02 | Shapes are normalized by peak and no feature describes their volume | At equal duration and factor the flooded volume differs up to 13.3 times between shapes |
| D-04 / H-06 | The flooding threshold (1.0 m³) coincides with the `.rpt` resolution (1 m³) | Between 27% and 57% of the overflowing nodes are labeled 0; 1,902 rows are exactly 1.0 |
| D-02 / H-04 | The DB contains no SWMM evidence (continuity error NULL in 1000/1000 runs) | The mass balance of the simulations cannot be demonstrated from the stored data |

**Test suite status:** 881 tests, 880 pass, 1 fails because of the environment (tkinter/Tcl, see 3.2).

---

## 2. Scope, method and limits

**Scope covered:** the whole `swmm_resilience/` package (~17,400 lines), `main.py`, `config*.yaml`, `scripts/`, `tests/`, the main documentation and that in `docs/`, and the real data in `outputs/` and `data/`.

**Method.**
- Five auditors in parallel, one per area: hydraulics (Fable), ML and data and traceability (Opus), analysis/visualization (Sonnet). Each wrote their annex with reproducible evidence.
- Read-only empirical checks: ~150 control SWMM simulations on copies in the scratchpad, SQLite databases opened with `mode=ro`, regeneration of out-of-fold XGBoost predictions (182 s, one family), 26 mutation tests on a copy of the repo.
- Independent verification by the consolidator of the critical findings and of several major ones (section 3).

**Declared limits.**
- **This branch** is audited. The worktree `.claude/worktrees/optuna-tuning-familia-ganadora` has 69 unmerged commits (phases B1, B2, C1, C2, Optuna, `tabular_v4_21` contract); its implementation was not verified (T-01, T-02).
- **Only XGBoost** was regenerated. The results of the other four families (random forest, linear, SVM, MLP) were not re-run; the arguments about them are from code.
- **Temporal LSTM/CNN surrogate:** there is no database or artifact in the tree; M-16 is code reading only.
- **Not verified:** running GUI, legacy pipeline with a populated DB, packaging test in a clean environment, the unit of the `.inp` coordinates, actual continuity errors of the 1000 training simulations (they are not stored), existence of the training `.rpt` files, `-m scale` tests.
- A minor incident: an auditor ran `python -m swmm_resilience.ml.train --help`, which has no argparse and attempted to train. It aborted for lack of `dataset_ml.csv` without writing anything; this was checked with `git status` and the dates of `outputs/models`. It was recorded as finding T-04.

---

## 3. Independent verification of the key findings

To avoid carrying unchecked claims into the report, the most important findings were re-verified by a route different from the auditor's.

### 3.1 Result

| Finding | How it was verified | Result |
|---|---|---|
| M-01 baseline beats XGBoost | Own script: leave-one-factor-out linear interpolation per (shape, node) on `dataset_final.csv` | **Confirmed.** RMSE 9.0 m³, F1 0.972 (identical to the auditor's) |
| M-02 volume with new shapes | Recomputed from the out-of-fold predictions parquet and the PreparedDataset targets | **Confirmed.** Oracle NSE: GroupKFold5 0.9553 · LOSO 0.9530 · LeaveOneShapeOut 0.8141. Error per shape −63.3% to +216.3%. `shape_quasi_uniform_block` = 61% of the SSE |
| M-03 per-fold mean with `zero_division=0` | Reading of `ml/bench/metrics.py:41-43` and `_average`; output of `outputs/metrics/metrics_classifier.json` | **Confirmed.** Published F1 0.74077 = mean of 25 folds, 5 of them with no positives (factors 0.2–1.0) with F1 fixed at 0. Pooled F1 0.9656 |
| M-07 / V-04 model ≠ promotion | `sha256sum` of `outputs/models/*.joblib` against `promotion.json` | **Confirmed.** `classifier.joblib` = `660be918…`; `promotion.json` declares `e2e7925e…` |
| M-08 / T-19 the bench discards the predictions | `main.py:198`: `_, metrics = evaluate_candidate(...)` | **Confirmed** |
| D-04 quantized volumes | Count over `dataset_final.csv` | **Confirmed with correction.** 46,236 of 46,242 positives are integers (the auditor said 100%). 1,902 are exactly 1.0 |
| H-01 fixed `InitFlow` | Reading of `[CONDUITS]` in the base `.inp` | **Partially confirmed.** The conduits have InitFlow ≠ 0 (5.5; 22; 22; 38; 50; 59 …) with `DYNWAVE`. The quantified impact (855 m³, up to 12.4%) is the auditor's and was not re-simulated |
| V-07 `halve_timeseries.py` | Reading | **Confirmed.** No `__main__` guard; it writes `DST` (the base Qx1 `.inp`) on import (line 35) |
| T-04 no argparse | Reading of `ml/train.py` | **Confirmed** (only `if __name__ == "__main__"`) |
| Confusion matrices | Recomputed from the parquet | **Confirmed** (section 6) |
| Repo integrity after the audit | `git status`, dates of `outputs/models` | Tree identical to the initial one except `docs/auditoria/` |

### 3.2 Tests

Full `pytest` on the current tree: **881 tests, 880 pass, 1 fails.** The failure is environmental: `TclError: couldn't read file ".../pythoncore-3.12-64/tcl/tcl8.6/auto.tcl"`. It is intermittent: in a first run (with `-x`) `test_display_image_shows_selected_map` failed and in the second `test_refresh_tree_populates_children`, both in `tests/desktop/test_results_tab.py`. Neither is a failure of project logic (V-34).
Additionally: `tests/database` 208 passed (auditor 3) and `tests/ml/bench` 51 passed (auditor 2).

### 3.3 Not re-verified by the consolidator

They are accepted with the evidence in the annex, but were not reproduced separately: H-02 to H-14 (except where indicated), D-01, D-02, D-03, D-05…D-22, M-04 to M-22, V-01 to V-06 and V-08 to V-34, T-01 to T-19 and the result of the 26 mutations. Each annex includes the scripts to reproduce them.

---

## 4. Consolidated findings matrix

**Scale.** 🔴 CRITICAL: compromises scientific validity · 🟠 MAJOR: biases results or produces incorrect comparisons · 🟡 MODERATE: limitation or latent risk · 🔵 MINOR: quality.

| Area | 🔴 | 🟠 | 🟡 | 🔵 | Total |
|---|---|---|---|---|---|
| Hydraulics (H) | 0 | 2 | 6 | 6 | 14 |
| ML and protocol (M) | 2 | 5 | 9 | 6 | 22 |
| Data and SQL (D) | 0 | 4 | 9 | 9 | 22 |
| Visualization and support (V) | 0 | 8 | 18 | 8 | 34 |
| Governance and documentation (T) | 0 | 6 | 6 | 7 | 19 |
| **Total** | **2** | **25** | **48** | **36** | **111** |

*(The data auditor reported 21 in their message; the annex lists 22: 4 major, 9 moderate and 9 minor.)*

### 4.1 Critical and major

| ID | Sev. | Finding | Impact | Annex |
|---|---|---|---|---|
| M-01 | 🔴 | LOSO/GroupKFold5 group by factor: they only measure interpolation. No baselines | No-learning baseline: RMSE 9.0 vs 68.4 m³ | 02 |
| M-02 | 🔴 | Shape features without volume; volume fails with unseen shapes | NSE 0.814; error per shape −63% to +216% | 02 |
| M-03 | 🟠 | F1/precision/recall averaged per fold with `zero_division=0` | F1 0.74 reported vs 0.966 actual | 02 |
| M-04 | 🟠 | Ranking by unweighted mean of per-fold RMSE, without CI | LeaveOneShapeOut: mean 79.7 vs pooled 136.1 m³; 95% CI [66.7, 212.2]. Differences of ≈6% between families fall within the noise | 02 |
| M-05 | 🟠 | Underfit regressor and `expm1(log1p)` retransformation bias | Train RMSE 118 ≈ OOF 127 m³; total volume −6%; small volumes +35%/+49% | 02 |
| M-06 | 🟠 | Families compared with fixed hyperparameters inherited from the previous winner | RF `max_depth=6`: NSE 0.45. The ranking compares configurations, not methods | 02 |
| M-07 | 🟠 | Default pipeline overwrites `outputs/models`; `promotion.json` false | SHA of the model on disk ≠ promoted SHA | 02 |
| H-01 | 🟠 | Fixed `InitFlow` (sum of peaks at factor 1) in the 1000 runs | 855 m³ at t=0; continuity up to 12.4%; +0.7 to +10.5% of flooded volume in short events at f=5 | 01 |
| H-02 | 🟠 | Shapes normalized by peak; no event-volume feature | ∫q_norm varies 72×; flooded volume up to 13.3× at equal duration and factor | 01 |
| D-01 | 🟠 | Provenance trigger only blocks `status`/`network_id` of a used run | Swapping `scenario_id` relabeled 6 of 96 volumes without warning (synthetic DB) | 03 |
| D-02 | 🟠 | Runs marked `COMPLETE` from the CSV, without SWMM evidence | `continuity_error_pct` NULL in 1000/1000; 988 with start=end timestamps | 03 |
| D-03 | 🟠 | Model provenance empty (`{}`, `fit_seconds=0`); contract hash over literal text | Models are trained from the CSV, not from the declared SQL view | 03 |
| D-04 | 🟠 | Threshold 1.0 m³ = `.rpt` resolution | 72 cases of "stops flooding when the factor increases", all with 1–2 m³: rounding noise | 03 |
| V-01 | 🟠 | Resilience, volume and factor figures evaluate in-sample without labeling | 25 factors × 160 training nodes | 05 |
| V-02 | 🟠 | `format_node_label` is not injective (`2C` and `2I` → "2") | 101 of 161 nodes share a label | 05 |
| V-03 | 🟠 | Legacy tools use the pre-v17 schema on the v17 DB | `create_schema` added 6 tables to a copy and then failed | 05 |
| V-04 | 🟠 | `predict_network` loads models without verifying provenance | Sept. figures generated with models different from the current ones | 05 |
| V-05 | 🟠 | `eda.py` uses legacy targets; on the current dataset the targets enter as features | Correlation and PCA contaminated; table vs target empty | 05 |
| V-06 | 🟠 | In the GUI, table and map of the same button can use different models | | 05 |
| V-07 | 🟠 | `halve_timeseries.py` overwrites the base `.inp` on import | It would invalidate the network hash of the models | 05 |
| V-08 | 🟠 | Reproduction scripts require Python 3.13; the environment is 3.12.10; `requirements.txt` without `build` | | 05 |
| T-01 | 🟠 | Phase A exists only as uncommitted changes | Risk of loss and divergence | 04 |
| T-02 | 🟠 | 69 commits in an unmerged worktree; an untracked document describes a contract that does not exist in this branch | | 04 |
| T-03 | 🟠 | 8 of 26 mutations are not detected by the tests | `tiempo_al_pico_h` at inference, nodes with base 0, `drain_down_hours`, junction coverage, ×1000 conversion of the `.rpt`, among others | 04 |
| T-04 | 🟠 | `python -m swmm_resilience.ml.train --help` runs the training | The README recommends it | 04 |
| T-05 | 🟠 | If `FLOW_UNITS` is missing from the `.inp`, the guard passes silently (SWMM assumes CFS) | Unit error of 28.3× | 04 |
| T-06 | 🟠 | Inference does not validate the feature contract that training does use | A required NaN would be imputed without warning | 04 |

> **Correction note (2026-10-11, spike D-04):** in the D-04 row, attributing the 72 cases to "rounding noise" is only partially correct. 12 were re-simulated: in all 12 the continuous volume goes down, in 7 with drops ≥ 0.3 m³ and in 2 with drops < 0.1 m³ (borderline cases). The other 60 were not verified. See `correcciones/03-resolucion-de-la-etiqueta.md` §4.1.

### 4.2 Moderate (summary; detail in the annexes)

| Area | IDs and topic |
|---|---|
| Hydraulics | H-03 uniform and synchronous training vs inference with arbitrary CSVs · H-04 continuity error not stored · H-05 6 h "drain" is a ramp (+24.5% of input volume in `shape_2h`) · H-06 label with 1 m³ resolution · H-07 fallback parser reads "Ponded Depth" (all labels → 0) · H-08 capacity only upstream, not of the outlet conduit |
| ML | M-08 the bench persists neither OOF nor provenance · M-09 implicit 0.5 threshold, uncalibrated probabilities, SVM `predict` ≠ `predict_proba` in 2.5% · M-10 confusion matrix, MCC, PR-AUC, per-shape breakdown missing · M-11 few independent units, no uncertainty · M-12 shape ablation with LOSO by factor; SHAP assumes tree · M-13 `--evaluate-shapes` evaluates on training · M-14 median imputation of `upstream_capacity_lps` in 57,000 rows (35.6%) · M-15 non-nested Optuna tuning · M-16 temporal surrogate with train/inference mismatch |
| Data | D-05 a read applies migrations without backup · D-06 `--persist-sql` neither atomic nor idempotent · D-07 `WorkflowLock` almost unused · D-08 `reset_db` can delete homonymous v17 tables · D-09 two incompatible schemas with homonymous tables · D-10 network hash depends on `autocrlf` · D-11 canonical data unversioned and without manifest · D-12 scenario hash does not cover the hydrograph · D-13 cutover phase 2 not implemented |
| Visualization | V-09…V-26: fixed "XGBoost" labels, nonexistent DB viewer, missing filters, unconditioned metrics, binary "resilience", 49 stacked node positions, log axes with zeros, swallowed exceptions, EDA plan with fixed "completed" steps, GUI thread safety, `parse_numeric_values` with decimal comma, test quality, dependencies, outdated Obsidian graph, GUI without the 5-family bench, fixed "(Qx1)" title |
| Governance | T-07 no document mentions `--bench*` or LeaveOneShapeOut · T-09 `config_7shapes.yaml` would revert Phase A · T-10 environment drift (Python 3.12/3.13/3.11; `optuna` not in requirements) · T-11 `scenario_predict.py` misclassified as pipeline A · T-12 Optuna spec contradicts the Phase A spec · T-13 ~69 MB of versioned `.pptx` |

The 36 minor findings are in the annexes.

---

## 5. What is fine and was verified

- **Labels:** independent re-simulations reproduce the v17 DB to the m³ (11 shape/factor combinations).
- **Hydrographs:** rescaling preserves volume (≤ 0.08%); at f=1 the pipeline reproduces the original `.inp`; Qx2/Qx3 are equivalent to Qx1×2 and ×3.
- **Topology and capacity:** Manning within ±3.7% of the SWMM capacity; slopes and offsets correct; the network is an acyclic tree (the `q_pico_acum` computation is valid here).
- **Train/serve skew:** dynamic features are computed the same way in training and inference (with the caveat H-03).
- **Data:** `integrity_check` clean in both DBs, no FK violations; 40×25×160 coverage complete; DB identical to `dataset_final.csv`; the 19-shape DB is an exact subset of the 40-shape one; network, model and contract hashes match; the bench manifest points to the current DB.
- **XGBoost reproducibility:** the bench and the legacy pipeline give digit-for-digit identical LOSO metrics (seed 42).
- **Comparison metrics:** `analysis/model_comparison.py` matches sklearn; `torch.load` uses `weights_only=True`; `factor_comparison.py` fails with an explicit error on inconsistent nodes.
- **DB and bench tests:** 208 and 51 pass.

---

## 6. Next step: confusion matrix of the chosen model(s)

### 6.1 Current state (findings M-08, M-10, T-19)

- `evaluate_candidate()` already produces a prediction per row (label, probability, volume), but `main.py:198` discards it and the reports only store per-fold means, from which a confusion matrix **cannot be reconstructed**.
- The only thing persisted per row is the `oof_predictions` table in SQLite, only for XGBoost with GroupKFold5 by factor (TN=111,468, FP=2,290, FN=920, TP=45,322). It differs slightly from the matrix regenerated in 6.2 (TN=111,431, FP=2,327, FN=877, TP=45,365); the origin of that difference was not investigated, and it does not change any conclusion.
- Missing in `ml/bench/metrics.py`: confusion matrix, specificity, MCC/balanced accuracy and PR-AUC.

### 6.2 Reference figures (XGBoost, current configuration, 40 shapes, 160,000 rows)

Out-of-fold predictions regenerated by the auditor with `evaluate_candidate()` and recomputed here. Positive class = `inunda` = 1, threshold 0.5, pooled matrix. **These are reference figures for a single family, not the final result of the thesis.**

| Protocol | TN | FP | FN | TP | Recall | Precision | Specificity | FN with vol ≤ 2 m³ |
|---|---|---|---|---|---|---|---|---|
| GroupKFold5 (by factor) | 111,431 | 2,327 | 877 | 45,365 | 0.9810 | 0.9512 | 0.9795 | 60.5% |
| LOSO (by factor) | 111,412 | 2,346 | 890 | 45,352 | 0.9808 | 0.9508 | 0.9794 | 62.7% |
| **LeaveOneShapeOut (by shape)** | 112,153 | 1,605 | 373 | 45,869 | 0.9919 | 0.9662 | 0.9859 | 88.2% |

Reading: in LeaveOneShapeOut no false negative exceeds 67 m³ and 88% have 2 m³ or less. Those cases are the ones in the 1–2 m³ band of the label, which by D-04 are unstable by construction: they must not be read as a model failure.

> **Correction note (2026-10-11, spike D-04):** "unstable by construction" is an exaggeration. In the re-simulated sample, the label is a deterministic function of the continuous volume, with no noise, with an effective threshold in (0.4988, 0.5002] m³. What was measured is that the `V_rpt = 1` band (continuous volume of ≈ 0.5 to 1.5 m³) is the closest to the threshold and concentrates 333 of the 380 FN. That this proximity explains the FN is an interpretation. See `correcciones/03-resolucion-de-la-etiqueta.md` §6.

### 6.3 Proposal

1. **Models:** one matrix for each chosen family (after tuning) and for each protocol, with **LeaveOneShapeOut as the main figure** and LOSO/GroupKFold5 only as an interpolation reference (M-01).
2. **Persist the out-of-fold predictions** of the bench (one parquet per candidate, together with the `prep_id`). This is the prerequisite: `main.py:198` must keep the first element of the `evaluate_candidate` tuple. It resolves M-08 and T-19 at once.
3. **Add to `metrics.py`:** `confusion_matrix`, specificity, MCC, PR-AUC and per-shape breakdown.
4. **Report:**
   - the pooled matrix (counts and normalized by actual row) as the main figure;
   - the per-fold dispersion (minimum, median, maximum) and a 95% CI by *group* bootstrap as uncertainty;
   - recall and precision per fold with **NaN** (not 0) when the fold has no positives (fixes M-03);
   - a second matrix separating the borderline band (actual volume of 1 to 2 m³) from the rest, per D-04/H-06;
   - the per-shape matrix for the out-of-range shapes (`shape_ultra_short_impulse`, F1 0.844; `common_bogota`, 0.906).

   > **Correction note (2026-10-11, spike D-04):** with the 1 m³ resolution, the only reported volume in [1, 2) m³ is the integer 1, which corresponds to a continuous volume of ≈ 0.5 to 1.5 m³. The per-band matrix is in `correcciones/03-resolucion-de-la-etiqueta.md` §6.1.

5. **Threshold sweep** from 0.2 to 0.8 as a sensitivity analysis, but with the operating threshold chosen **in an inner CV**, never on the predictions that are reported.

### 6.4 Decisions you have to make

| # | Decision | Recommendation |
|---|---|---|
| 1 | Which models are included? (only the winner after Optuna or the 5 families) | The 5 families in the annex; the winner in the body of the thesis |
| 2 | Main protocol? | LeaveOneShapeOut |
| 3 | Decision threshold? | 0.5 as baseline and operating threshold chosen by inner CV |
| 4 | Is the 1–2 m³ band treated separately? | Yes, in a complementary matrix |
| 5 | Pooled or per-fold matrix as the main figure? | Pooled, with the per-fold dispersion as uncertainty |

---

## 7. Prioritized action plan

This is only a proposed order; nothing has been modified.

**P0 — before citing any result in the thesis**
1. State that LOSO/GroupKFold5 measure interpolation, add baselines and switch to LeaveOneShapeOut as the main protocol (M-01).
2. Fix the aggregation of F1/precision/recall (M-03) and report the pooled value with its dispersion (M-04, M-11).
3. Persist the out-of-fold predictions and build the confusion matrix (section 6).
4. Reconcile the model on disk with `promotion.json`, block the silent overwriting of `outputs/models` and regenerate the figures with the declared model (M-07, V-04).
5. Label as "in-sample" the figures and outputs that are (V-01, M-13).
6. Neutralize `halve_timeseries.py` and add argparse to `ml/train.py` (V-07, T-04).
7. Confirm the Phase A commit and decide the integration of the 69-commit worktree (T-01, T-02).

**P1 — scientific validity and traceability**
8. Decide how to represent the event volume in the features (M-02, H-02). It is a research decision, not a patch.
9. Measure the sensitivity to `InitFlow = 0` and to the 6 h "drain" with a subset of re-simulations (H-01, H-05).
10. Read the flooded volume with a higher resolution than the `.rpt` and review the threshold (H-06, D-04).

    > **Correction note (2026-10-11, spike D-04):** with the current `REPORT_STEP` (3 min), the integral of the `flooding` series of the `.out` is **not** a higher-resolution source: it departs from the continuous volume by up to 9.5 m³ per node, and by up to 0.89 m³ even at 5 s. The only measured source that gives the continuous volume is pyswmm's `node.statistics['flooding_volume']`. The spike recommends path A (declare, without re-simulating). See `correcciones/03-resolucion-de-la-etiqueta.md`.

11. Store the continuity error and the SWMM evidence in the DB (H-04, D-02, D-03).
12. Strengthen the provenance triggers and the atomicity of `--persist-sql` (D-01, D-06).
13. Close the 8 mutations without a test (T-03).

**P2 — maintenance**
14. Update the documentation (34 discrepancies, T-07), clean up orphans and `.gitignore`, pin the Python environment (T-08…T-17, V-08).
15. Migrate or retire the legacy tools that use the pre-v17 schema (V-03, D-09, D-08).

---

## 8. Traceability with the previous audits

Annex 04 traces **117 items** from the previous audits and the follow-up documents against the current code.

| Status | Items |
|---|---|
| Closed | 32 |
| Partial | 30 |
| Open | 40 |
| Not applicable | 13 |
| Not verified | 2 |

- Of the 18 technical findings of `AUDITORIA_SURROGATE_HIDROGRAMAS.md`: **4 closed** (M5, MO5, m2, m5), **11 partial** and **3 open** (M3 drain, MO1, MO4).
- Those listed as closed but with no test that would detect their regression are in T-03.
- Junk-code audit: `pruebas_locales.py`, `verificar.py` and `outputs/factor_comparisons/` were deleted. `halve_timeseries.py` is still versioned and dangerous (V-07). There are still no unreachable package modules. The audit misclassified `scenario_predict.py` as pipeline A (T-11).
- There are **34 documentation ↔ code discrepancies** (for example: README and DICCIONARIO say 15 features and there are 17; several documents say `--predict` does not run SWMM and it does; `COMANDOS.md` proposes a validation command that fails because of format).

---

## 9. Hydraulic assumptions to declare in the thesis

1. The network is simulated with **dynamic wave** (`DYNWAVE`), a 5 s routing step and an input hydrograph that is spatially **uniform and synchronous** at all nodes (H-03).
2. The initial conditions of the conduits (`InitFlow`) are those of the base `.inp` and do not change with the factor or the shape (H-01).
3. Flooding is the volume SWMM reports per node, with **1 m³ resolution**; the label is `volume ≥ 1 m³` (H-06, D-04).

   > **Correction note (2026-10-11, spike D-04):** in the re-simulated sample (one network, 17 simulated scenarios, 15 with overflow; the threshold bound was computed on the 15 with per-node record, 13 with overflow), "reported volume ≥ 1 m³" is consistent with an effective threshold of ≈ 0.5 m³ on the continuous volume, bounded to (0.4988, 0.5002] m³. Minor overflows count as "no flooding".

4. The final stretch of the hydrograph is extended with a zero-flow point 6 h after the last datum, which can create a ramp (H-05).
5. Shapes are normalized by peak; the model does not receive the event volume (H-02).
6. The capacity used is that of the **upstream** conduits (H-08).
7. The network is a tree with no cycles or dividers, so the sum of upstream peaks is valid (H-12).
8. The training labels have no stored continuity error (H-04, D-02).

---

## 10. Reproducibility of this audit

- Each annex includes its diagnostic scripts (all read-only). They are run with `venv\Scripts\python.exe` from the repo root; the DBs are opened with `mode=ro`.
- The out-of-fold predictions used in section 6 were regenerated with `evaluate_candidate()` and the `config.yaml` configuration on the PreparedDataset `af3751551ca5df98` (160,000 rows, 40 shapes). The script is in annex 02.
- Repo state at close: identical to the initial one, plus the `docs/auditoria/` folder. None of this is committed.
