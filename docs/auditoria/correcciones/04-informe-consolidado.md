# Correction 04 — Consolidated report of the audit corrections (steps 1 to 6)

- **Date:** 2026-10-11
- **Branch:** `worktree-optuna-tuning-familia-ganadora`, commits `b831f0f` … `867ce6e` plus the commit of this documentation. Series base: `9ad4df0`. No push (pending authorization).
- **Spec and plan:** [`../../superpowers/specs/2026-10-11-correcciones-auditoria-design.md`](../../superpowers/specs/2026-10-11-correcciones-auditoria-design.md), [`../../superpowers/plans/2026-10-11-correcciones-auditoria.md`](../../superpowers/plans/2026-10-11-correcciones-auditoria.md).
- **Source audit:** [`../2026-10-10-auditoria-completa.md`](../2026-10-10-auditoria-completa.md) and annexes 01–05.
- **Detail documents:** [`01-evaluacion-honesta.md`](01-evaluacion-honesta.md), [`02-trazabilidad-y-seguridad.md`](02-trazabilidad-y-seguridad.md), [`03-resolucion-de-la-etiqueta.md`](03-resolucion-de-la-etiqueta.md). Data: [`datos/`](datos/README.md).

**Scope warning for all the evaluation figures.** XGBoost with the default hyperparameters of `config.yaml` (not the model tuned with Optuna); a single network; `tabular_v4_21` contract; 154,080 rows = 963 of the 1000 runs (shape, factor) × 160 nodes, after the 5 % continuity filter (`datos/tables_provenance.json`; `datos/runs/provenance.json`).

---

## 1. Status per finding

| Finding | Sev. (audit) | Status | Commits | Document |
|---|---|---|---|---|
| M-01 — LOSO/GroupKFold5 measure interpolation; no references | 🔴 | **Corrected in the evaluation.** Two no-learning references in the benchmark; what each protocol measures is written in the reports; LeaveOneShapeOut declared as the main protocol; measured with `tabular_v4_21` | `9c8db1b`, `02cdaab`, `6c48c46`, `833760d`, `5f0586d`; measurement `a26155e`, `4a2a87c`, `4717885`, `395a2ee`, `e26fc5a`, `284927d` | [01 §1](01-evaluacion-honesta.md) |
| M-02 — volume generalizes poorly to new shapes | 🔴 | **Measured; partly mitigated; open.** The `tabular_v4_21` contract (prior to this series) improves the oracle NSE, but one shape concentrates 71.6 % of the squared error | `b831f0f`, `bd0f23f`; measurement (same) | [01 §2](01-evaluacion-honesta.md) |
| M-03 — F1 averaged per fold with zeros | 🟠 | **Corrected in the benchmark.** `classifier` is the pooled value; the previous one is kept under an explicit name. `by_factor` and the legacy evaluator do not change | `b831f0f`, `2871917`, `bd0f23f`, `5528479` | [01 §3](01-evaluacion-honesta.md) |
| M-08 / T-19 — the benchmark does not persist out-of-fold predictions | 🟡 | **Corrected.** `--bench-evaluate` and the `tabular_v4_21` script save per-row predictions and provenance | `b95a480`, `4a2a87c`, `e26fc5a` | [01 §4](01-evaluacion-honesta.md) |
| M-10 — confusion matrix and other metrics missing | 🟡 | **Partial.** Confusion matrix, specificity, per-shape breakdown, MAE and pooled total volume error added. MCC, balanced accuracy and PR-AUC missing | `b831f0f`, `bd0f23f` | [01 §4](01-evaluacion-honesta.md) |
| V-01 (and M-13 in part) — in-sample figures without labels | 🟠 | **Corrected for the new curve and per-factor comparison figures** and for `--evaluate-shapes`. Existing figures not regenerated; `--evaluate-generalization` and `--evaluate-validation-shapes` without a label | `93e34b4` | [02 §1](02-trazabilidad-y-seguridad.md) |
| V-07 — `halve_timeseries.py` overwrites the base network | 🟠 | **Corrected** | `57dc170` | [02 §2](02-trazabilidad-y-seguridad.md) |
| M-07 / V-04 — model ≠ `promotion.json`; figures without provenance | 🟠 | **Corrected in `predict_network` and in the figures of `main.py` and `runner.py`.** Other loaders remain unverified | `8f17147`, `0c8a3f4` | [02 §3](02-trazabilidad-y-seguridad.md) |
| New defect: tests that wrote to the real `outputs/metrics` | — | **Corrected** (isolated tests and automatic guard) | `398d6bc` | [02 §4](02-trazabilidad-y-seguridad.md) |
| D-04 / H-06 — label resolution | 🟠 / 🟡 | **Investigated (spike). Path A/B decision pending for the author.** Recommendation: path A | `2035a51`, `867ce6e` and the commit of this documentation (two minor corrections) | [03](03-resolucion-de-la-etiqueta.md) |

---

## 2. Before and after of the key metrics

"Before" = audit figures, measured with the `tabular_v3_17` contract (17 features) on 160,000 rows without continuity filter (PreparedDataset `af3751551ca5df98`; annex 02). "After" = this series, `tabular_v4_21` contract on 154,080 rows. **The contract and the set of rows change at the same time: it is not a controlled experiment.** The table describes two states of the project; the differences are not attributed to a specific change. Source abbreviations: A02 = `../anexos/02-ml-protocolo-evaluacion.md`; AC = `../2026-10-10-auditoria-completa.md`; CP = `datos/comparacion_protocolos.csv`; MC = `datos/matriz_confusion.csv`; PF = `datos/por_forma.csv`.

| Metric | Model | Protocol | Before | Source before | After | Source after (column) |
|---|---|---|---|---|---|---|
| Pooled F1 | XGBoost | LOSO | 0.9656 | A02 M-03 | 0.9667 | CP `f1_agregado` |
| Zero-filled per-fold mean F1 (previous definition) | XGBoost | LOSO | 0.7408 | A02 M-03 | 0.7379 | CP `f1_media_por_fold_con_ceros` |
| Pooled F1 | XGBoost | LeaveOneShapeOut | 0.9789 | A02 M-01 | 0.9817 | CP `f1_agregado` |
| Pooled end-to-end RMSE (m³) | XGBoost | LOSO | 68.43 | A02 M-01 | 57.89 | CP `rmse_vol_agregado_m3` |
| Per-fold mean end-to-end RMSE (m³) | XGBoost | LOSO | 55.3 | A02 M-04 | 46.28 | CP `rmse_vol_media_por_fold_m3` |
| Pooled end-to-end RMSE (m³) | XGBoost | LeaveOneShapeOut | 136.1 | A02 M-04 | 106.21 | CP `rmse_vol_agregado_m3` |
| Per-shape mean end-to-end RMSE (m³) | XGBoost | LeaveOneShapeOut | 79.7 | A02 M-04 | 56.81 | CP `rmse_vol_media_por_fold_m3` |
| Oracle regressor NSE | XGBoost | LeaveOneShapeOut | 0.8141 | AC §3.1 | 0.8906 | CP `nse_regresor_oracle` |
| Oracle regressor NSE | XGBoost | LOSO | 0.9530 | AC §3.1 | 0.9675 | CP `nse_regresor_oracle` |
| Oracle regressor NSE | XGBoost | GroupKFold5 | 0.9553 | AC §3.1 | 0.9694 | CP `nse_regresor_oracle` |
| Total vol. error per shape (min.; max.) | XGBoost | LeaveOneShapeOut | −63.3 %; +216.3 % | A02 M-02 | −52.1 %; +33.3 % | PF `error_vol_pct` |
| Share of the squared error of the worst shape (`shape_quasi_uniform_block`) | XGBoost | LeaveOneShapeOut | 61.35 % | A02 M-02 | 71.6 % | Derived from PF (`rmse_vol² × n_rows`) |
| Pooled end-to-end RMSE (m³) | Factor interpolation | LOSO | 9.00 | A02 M-01 | 9.13 | CP `rmse_vol_agregado_m3` |
| Pooled F1 | Factor interpolation | LOSO | 0.9718 | A02 M-01 | 0.9718 | CP `f1_agregado` |
| Pooled end-to-end RMSE (m³) | Nearest shape | LeaveOneShapeOut | 170.2 | A02 M-01 | 172.86 | CP `rmse_vol_agregado_m3` |
| Pooled F1 | Nearest shape | LeaveOneShapeOut | 0.9758 | A02 M-01 | 0.9764 | CP `f1_agregado` |
| Perfect predictor, zero-filled per-fold mean F1 / pooled | — | LOSO | 0.80 / — | A02 M-03 | 0.8000 / 1.0000 | Command of [01 §5](01-evaluacion-honesta.md) |

**XGBoost confusion matrices (TN / FP / FN / TP).**

| Protocol | Before (160,000 rows; AC §6.2) | After (154,080 rows; MC) |
|---|---|---|
| GroupKFold5 | 111,431 / 2,327 / 877 / 45,365 | 105,711 / 2,144 / 844 / 45,381 |
| LOSO | 111,412 / 2,346 / 890 / 45,352 | 105,610 / 2,245 / 879 / 45,346 |
| LeaveOneShapeOut | 112,153 / 1,605 / 373 / 45,869 | 106,526 / 1,329 / 380 / 45,845 |

The "before" matrices come from predictions regenerated by the audit in a temporary folder; they were not persisted. The "after" ones are reconstructed from files with a recorded SHA-256 (`datos/tables_provenance.json`).

The only **controlled** comparison between contracts (same rows) is that of the per-shape mean RMSE in LeaveOneShapeOut: 78.0368 m³ (`tabular_v3_17`) versus 56.8088 m³ (`tabular_v4_21`), in `../../resultados/2026-10-06-explicacion-completa.md` §11.1. It was not re-run here; the run of this series reproduces 56.8088 (CP `rmse_vol_media_por_fold_m3`).

---

## 3. Current results: XGBoost versus the references in the three protocols

Source: CP (metrics) and MC (matrix). All on 154,080 rows.

| Protocol | What it measures | Model | Pooled F1 | Pooled RMSE (m³) | Per-fold mean RMSE (m³) | Total vol. error (%) | Oracle NSE | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LOSO | Interpolation between factors | Factor interpolation | 0.9718 | 9.13 | 4.77 | −0.39 | 0.9992 | 105,617 | 2,238 | 416 | 45,809 |
| LOSO | same | XGBoost | 0.9667 | 57.89 | 46.28 | −5.03 | 0.9675 | 105,610 | 2,245 | 879 | 45,346 |
| GroupKFold5 | same, 5 factors per fold | Factor interpolation | 0.9718 | 9.13 | 7.47 | −0.39 | 0.9992 | 105,617 | 2,238 | 416 | 45,809 |
| GroupKFold5 | same | XGBoost | 0.9681 | 56.19 | 55.44 | −4.64 | 0.9694 | 105,711 | 2,144 | 844 | 45,381 |
| **LeaveOneShapeOut** | **Unseen hydrograph shapes (main)** | Nearest shape | 0.9764 | 172.86 | 98.14 | −2.98 | 0.7102 | 106,556 | 1,299 | 889 | 45,336 |
| **LeaveOneShapeOut** | same | **XGBoost** | **0.9817** | **106.21** | **56.81** | −7.98 | **0.8906** | 106,526 | 1,329 | 380 | 45,845 |

**How to read it.**

- In LOSO and GroupKFold5 the no-learning reference wins: those protocols measure interpolation between factors. With this grid, GroupKFold5 gives the interpolation reference exactly the same predictions as LOSO (maximum difference 0.0), because no fold holds out two contiguous factors; checked by the implementer (report of task 8) and repeated for [01 §5](01-evaluacion-honesta.md).
- In LeaveOneShapeOut, XGBoost beats the nearest shape in volume and, narrowly, in F1; the reference has a smaller total volume error.
- The interpolation uses a fallback in 1.25 % of its rows and the nearest shape in 0.42 % (`datos/runs/provenance.json`, `baseline_fallback`).
- The ROC-AUC of the references is not interpreted (they do not produce probabilities).

---

## 4. Label resolution (D-04 / H-06): pending decision

Summary of [`03-resolucion-de-la-etiqueta.md`](03-resolucion-de-la-etiqueta.md):

- In the re-simulated sample (17 scenarios, 15 with overflow; per-node record in 15 scenarios, 13 with overflow), the label `V_rpt ≥ 1 m³` is equivalent to `V_continuo ≥ t` (continuous volume) with t in (0.4988, 0.5002] m³, with no discrepancies in 2,400 node-scenario cases: it is a deterministic quantization, not noise (03 §4).
- The integral of the `.out` with the current `REPORT_STEP` (3 min) is not a better source: it departs from the continuous volume by up to 9.5 m³ per node (03 §3.2).
- In LeaveOneShapeOut, 333 of the 380 false negatives of XGBoost have `V_rpt = 1` (`datos/banda_limitrofe.csv`); without that band, F1 goes from 0.9817 to 0.9847 (03 §6.2, only as a sensitivity analysis, with asymmetric exclusion).
- Re-simulating the 1000 runs would cost on the order of 17–20 min of SWMM, but would force redoing the dataset, the DB, the `prep_id`, the benchmark, Optuna, the validation and all the figures (03 §5).

**Spike recommendation: path A** (keep the resolution and the threshold, declare them as an assumption and report the matrix with the band separated). **The decision belongs to the author and is pending.** The thesis text about the label (03 §8.2) and the presentation of Table 4.2 of `01` depend on that decision: with path B the labels, the `prep_id` and all the figures of this report would change.

---

## 5. Limitations of the figures of this report

1. **Untuned XGBoost.** Default hyperparameters of `config.yaml` (200 trees, depth 6, learning rate 0.05, subsampling 0.8). The model tuned with Optuna was not evaluated against the references. The run uses its own script (`scripts/ejecutar_evaluacion_v4_21.py`) and not the benchmark CLI, so as not to overwrite the historical reports or `outputs/models` (decision log R1).
2. **A single network, 160 nodes.** No protocol holds out nodes or networks.
3. **963 of 1000 runs.** The continuity filter leaves out 37 (shape, factor) cells, concentrated in the short or extreme shapes (`shape_ultra_short_impulse` 10, `rare_extreme` 7, `common_bogota` 5; report of task 8). Those shapes are also the worst classified (`datos/por_forma.csv`).
4. **References with fallback.** 1.25 % (interpolation) and 0.42 % (nearest shape) of their rows do not use their primary source.
5. **The nearest-shape reference** defines "nearest" only by duration and time to peak, with descriptors taken from the `tabular_v3_17` dataset `2db0578f875526cb` (same 40 shapes; R7).
6. **Uncontrolled before/after comparisons** (§2).
7. **No confidence intervals.** The per-fold dispersion is reported, but not a CI by group resampling (M-04, M-11). The independent units are 25 factors and 40 shapes, not 154,080 rows.
8. **Error concentration.** The pooled RMSE of LeaveOneShapeOut depends 71.6 % on one shape; without it, it would be 57.32 m³ (derived from `datos/por_forma.csv`, [01 §2](01-evaluacion-honesta.md)).
9. **Label with 1 m³ resolution** (pending decision, §4).
10. **Per-row predictions not versioned** in git (`outputs/` ignored); they are reproduced and verified with the SHA-256 values of `datos/tables_provenance.json`.

---

## 6. What was not done and why

| Topic | Finding | Why it was not done | Where it stands |
|---|---|---|---|
| Fixed `InitFlow` in all runs | H-01 🟠 | Requires re-simulating a subset to measure sensitivity; outside the scope of steps 1–6 (spec). Its impact is the auditor's figure (annex 01) and it was not re-simulated | Audit plan, P1 item 9 |
| Event volume variables beyond `tabular_v4_21` | H-02 / M-02 | Out of scope by design (spec §7): decided after measuring, and the measurement is in `01` §2 | Research decision of the author |
| Ablation of the volume features under LeaveOneShapeOut | M-02, M-12 | Was not in the spec | — |
| Continuity error and SWMM evidence in the DB; model provenance in SQL | D-02, D-03, H-04 | Outside the scope of steps 1–6 | Audit plan, P1 item 11 |
| Verification of `promotion.json` in other loaders (`ScenarioPredictor`, SHAP, `predict_from_inp` and desktop) | M-07 / V-04 | Spec §2.8 limits the verification to `ml/predict.py` (R8) | `02` §3 |
| Regenerate the existing figures with the declared model | V-01, V-04 | The final model of the thesis is not fixed (Part 2) | — |
| Label in `--evaluate-generalization` and `--evaluate-validation-shapes` | M-13 | Spec §2.6 covered only `--evaluate-shapes` | — |
| Out-of-sample curves from the out-of-fold predictions | V-01 | Optional improvement (spec §7) | — |
| Main ranking metric (per-fold mean) and group CI | M-04, M-11 | Was not in the spec; `end_to_end_pooled` already gives the pooled value | — |
| `expm1(log1p)` retransformation bias; underfitted regressor | M-05 | Was not in the spec | — |
| Families with inherited hyperparameters; fair SVM; final XGBoost + MLP selection | M-06, M-09 (SVM) | It is **Part 2**, a separate subproject with its own spec | Part 2 |
| Explicit decision threshold and calibration | M-09 | Was not in the spec | — |
| Non-nested Optuna tuning | M-15 | Was not in the spec | Part 2 |
| `python -m swmm_resilience.ml.train --help` trains | T-04 | Was not in the spec, although the audit paired it with V-07 | — |
| Fallback parser that reads "Ponded Depth" | H-07 | Outside the scope of the spike (no code changes); recommended in `03` §8 | Independent of path A/B |
| GUI: table and map with different models | V-06 | Was not in the spec | — |
| Push to `origin` | — | Requested from the author at the end (R3) | — |

---

## 7. Tests

| Moment | Result | Source |
|---|---|---|
| Baseline (commit `1ecca50`) | 1,049 pass, 0 fail | Spec §1; commit `3f7a628` corrects an initial figure of 1,058 that counted dots of warning lines |
| After each task | 1,065 (T1) · 1,073 (T2) · 1,093 (T3) · 1,098 (T4) · 1,103 (T5) · 1,107 (T6) · 1,128 (T7) · 1,145 (R10) · 1,154 (R9) · 1,159 · 1,165 (T8) | Execution log (unversioned working notes) |
| **Close of the series (this documentation)** | **1,164 pass, 1 fails, 3 deselected (`scale`), 187 s.** The failure is `tests/desktop/test_results_tab.py::test_refresh_tree_populates_children` (`TclError: invalid command name "tcl_findLibrary"`), a Tcl/Tk environment failure known since the audit (AC §3.2). Re-running that file: **7 pass**. Total green: **1,165** (116 more than the baseline) | `../../../venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=<short temporary> -o addopts="-m 'not scale'"` |

---

## 8. Deferred nitpicks (review log)

They do not change any figure; they are recorded so as not to lose them.

| Task | Nitpick |
|---|---|
| T1 (metrics) | RMSE is computed in two ways (`np.sqrt(np.mean(...))` and `mean_squared_error`); `shape_breakdown_entry` with empty arrays emits a warning |
| T2 (`score_predictions`) | The routing test does not exercise it (all classifications are correct); the name of a parity test promises more than it checks; `score_predictions` and `pooled_classifier_metrics` duplicate the concatenation and the counts; `protocols` is not validated against the dataset (a misspelled name or a bare string); the reasoning of a docstring of the parity test was lost |
| T3 (references) | The interpolation tests mostly use midpoints (no irregular grid or specific GroupKFold5 test); no test detects standardizing with statistics of the test shape; weak assertion `match="C"`; `inf` is not checked in the descriptors; the duplicate/NaN messages of the nearest shape have no test; the fallback counters are totals over protocols (documented); use of `assert` (skipped with `-O`) |
| T4 (persistence) | The CLI test does not re-read the contents of the parquet/JSON of the references; `metrics_<familia>.json` carries no `prep_id` and may sit next to a newer `provenance.json`; mid-file imports in `test_reports.py`; no NaN→null test in the provenance; no test that another `ValueError` propagates |
| T5 (in-sample) | The test checks the text of the note, not its position; the source-code check is anchored to the first `if args.evaluate_shapes:`; with more than 30 nodes the note sits close to the "Node ID" label |
| T6 (`halve_timeseries`) | Loose assertions (`in`); `main()` without a test; race window between checking and writing; no warning if there are no rows to divide |
| T7 (promotion) | Malformed `promotion.json`/`manifest.json` give `JSONDecodeError` without a domain message; provenance granularity per folder; the `outputs/metrics` guard has no test of its own; an empty folder `outputs/metrics/factor_comparison/` remains |
| T8 (run and tables) | The inner functions of `main()` of the run script have no test; the docstring about `ensure_allowed_output` promises more than it does; `informe_evaluacion.py --out` has no guard; `runs/*.json` lack the `n_runs`/`git_dirty` fields (declared in `datos/README.md`); `git_dirty` ignores untracked files (definition written in the JSON) |
| T9 (spike) | Prudent claims but not supported by measurement: the regressor bias as regression to the mean (03 §6.4), the closeness of the FPs to the boundary (03 §6.2), the design conclusion of 03 §7.1 and the wording of 03 §3; the spike commit carries the signature of a model other than the intended one (note only) |

---

## 9. Follow-up: pending corrections in other documents

This commit corrects in `03` the scenario count (17 simulated, 15 with overflow; 15 with per-node record, 13 with overflow) and the quantization bound of §7.6 (≤ 0.51 m³, due to node `8C`, which differs by 0.508 m³). The same figures appeared in dated notes of other documents. **The first four rows were applied in the final correction wave** (text of the notes corrected to "17 simulated scenarios, 15 with overflow" and "≤ 0.51 m³"); the last two remain untouched on purpose:

| Document and line | Current text | Suggested correction |
|---|---|---|
| `../2026-10-10-auditoria-completa.md`, l. 275 (note of assumption 3) | "in the re-simulated sample (one network, 13 scenarios with overflow)" | "17 simulated scenarios (15 with overflow); the threshold bound was computed on the 15 with per-node record (13 with overflow)" |
| `../anexos/03-datos-persistencia-sql.md`, l. 76 (note on D-04, point 1) | same | same |
| `datos/README.md`, l. 22 | same | same |
| `../anexos/01-hidraulica-swmm.md`, l. 113 (note of H-06) | "(≤ 0.5 m³ below 100 m³)" | "(≤ 0.51 m³)" |
| `../2026-10-10-auditoria-completa.md`, l. 215 (decision 4, "1–2 m³ band") | Without specifying that it is only the integer 1 | Already listed as not applied in 03 §8.1 |
| `03-resolucion-de-la-etiqueta.md`, Appendix, `s09_notas.py` | Contains the original text of the notes | Left as is: it is the record of the script that was executed |

Pending after the final wave: the `que_mide` column of `datos/comparacion_protocolos.csv` keeps the previous text of `PROTOCOL_INTERPRETATION["GroupKFold5"]` ("5 bloques retenidos", i.e. "5 held-out blocks"), because the CSV is hash-traced in `tables_provenance.json` and was not regenerated. The current wording is "Interpolación entre factores en 5 grupos de factores intercalados; ningún pliegue retiene dos factores contiguos; todas las formas siguen presentes en el entrenamiento." (i.e. "Interpolation between factors in 5 groups of interleaved factors; no fold holds out two contiguous factors; all shapes remain present in training."). Regenerate the CSV and `tables_provenance.json` with `scripts/informe_evaluacion.py` when the run is executed again.

---

## 10. Decisions taken during execution (summary)

From the execution log (unversioned), those that affect the reading of the figures:

| Id | Decision | Consequence |
|---|---|---|
| R1 | The measurement on real data uses `scripts/ejecutar_evaluacion_v4_21.py`, not the CLI, with the default hyperparameters of `config.yaml` | The figures describe untuned XGBoost; `outputs/models`, `outputs/bench/reports` and `ranking.csv` were not touched |
| R7 | The nearest-shape reference receives explicit shape descriptors, taken from the `tabular_v3_17` dataset | "Nearest" = by duration and time to peak |
| R9 | The references work on the incomplete grid (963 of 1000 runs) instead of re-running on the 160,000 unfiltered rows | References and XGBoost are scored on exactly the same rows; 1.25 % and 0.42 % fallback rows |
| R10 | Isolate the tests that wrote to `outputs/metrics` and add a guard | See `02` §4 |
| R11 | The fallback counter of the interpolation is defined against the training grid of the fold | The factors deliberately held out by GroupKFold5 do not count as fallback |
| R13 | The spike added dated notes to the earlier documents without rewriting the original text | See `03` §8.1 and §9 of this report |

---

## 11. Summary for the research report

> **Evaluation of the surrogate model after the protocol audit**
>
> An internal audit of the evaluation protocol identified three problems that affected the reading of the results: two of the cross-validation protocols measured interpolation and not generalization, no-learning references to compare against were missing, and the classification metrics were averaged per fold in a way that underestimated them. The evaluation and its traceability were corrected, and the measurement was repeated with the 21-variable contract, which includes integrated variables of the event volume. The following figures correspond to XGBoost with default hyperparameters (not to the final tuned model), to a single drainage network of 160 nodes and to 963 of the 1000 simulations (hydrograph shape, intensity factor), those that meet a continuity error of at most 5 %, that is, 154,080 node-scenario cases.
>
> **What each protocol measures.** When a complete intensity factor is held out (LOSO) or five factors per fold (GroupKFold5), a reference that only interpolates the volume between the neighbouring factors of the same shape and node obtains a volume RMSE of 9.13 m³, versus 57.89 m³ for XGBoost in LOSO, and an F1 of 0.9718 versus 0.9667. Those protocols measure interpolation between already seen factors and are not used as evidence of generalization. With the 25-factor grid, GroupKFold5 is not more demanding than LOSO, because no fold holds out two contiguous factors.
>
> **Main protocol: unseen hydrograph shape (LeaveOneShapeOut).** XGBoost beats the reference that copies the volume of the most similar training shape: pooled RMSE of 106.21 versus 172.86 m³, NSE of the volume estimated on the nodes that actually flooded of 0.8906 versus 0.7102 and F1 of 0.9817 versus 0.9764. The pooled confusion matrix of XGBoost has 45,845 true positives, 106,526 true negatives, 1,329 false positives and 380 false negatives; 333 of those false negatives are nodes with the minimum volume that the label considers flooding (1 m³ reported) and none exceeds 100 m³.
>
> **The magnitude of the volume is the weak part.** The error of the total volume per shape ranges from −52.1 % to +33.3 %, and a single shape, with an almost uniform block inflow, concentrates 71.6 % of the squared error, with its volume underestimated by 52.1 %. Without that shape, the pooled RMSE of the other 39 would be 57.32 m³. With the previous 17-variable contract, the audit had measured an NSE of 0.8141 and a per-shape error of −63.3 % to +216.3 %, on a different set of rows; the comparison does not isolate the effect of the new variables.
>
> **Metric aggregation.** The classification metrics are computed on the pooled confusion matrix. Averaging per fold while assigning F1 = 0 to the folds without flooding reduced the LOSO F1 from 0.9667 to 0.7379, and gave 0.80 to a perfect predictor; the apparent difference between protocols was an effect of that aggregation.
>
> **Traceability.** The out-of-fold predictions are saved per row with the SHA-256 fingerprint of each file, so that all the tables can be reconstructed without retraining; the figures evaluated on training data are labelled as in-sample fit, and prediction with the promoted model stops if the promotion record exists (`promotion.json`, versioned together with the models) and its files do not match it; if the record is missing, it only warns.
>
> **Pending.** The flooding label uses the volume of the SWMM report, with a resolution of 1 m³; in control re-simulations it is equivalent to a threshold of ≈ 0.5 m³ on the continuous volume. Whether to keep that definition and declare it as an assumption (recommended option) or to re-simulate the 1000 runs is a decision still open; the second would change all the figures above. The tuned model and the other families have also not yet been evaluated with this protocol.

**Sources of the summary figures:** `datos/comparacion_protocolos.csv` (F1, RMSE, NSE), `datos/matriz_confusion.csv` (matrix), `datos/banda_limitrofe.csv` (333 FN, 0 FN ≥ 100 m³), `datos/por_forma.csv` (per-shape range, 71.6 %, 57.32 m³; derived in [01 §5](01-evaluacion-honesta.md)), `datos/tables_provenance.json` (154,080 rows, 963 of 1000), annex 02 (0.8141; −63.3 % to +216.3 %), [01 §5](01-evaluacion-honesta.md) (perfect predictor 0.80), [03](03-resolucion-de-la-etiqueta.md) (≈ 0.5 m³).
