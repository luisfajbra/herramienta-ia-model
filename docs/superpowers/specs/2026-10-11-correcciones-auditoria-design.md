# Audit corrections (steps 1 to 6) — design

**Date:** 2026-10-11
**Branch:** `fix/cierre-auditoria-hidrogramas-20260927` (checked out in the `optuna-tuning-familia-ganadora` worktree).
**Source:** `docs/auditoria/2026-10-10-auditoria-completa.md` and annexes 01–05.
**Scope:** findings M-01, M-02, M-03, V-01, V-07, M-07/V-04 and D-04/H-06, plus the persistence of out-of-fold predictions (M-08/T-19). Part 2 (fair SVM and final XGBoost + MLP selection) is a separate subproject with its own spec.

## 1. Verified starting state (re-triage in the worktree)

| Finding | State in the worktree |
|---|---|
| M-01 no no-learning references | Open. `LeaveOneShapeOut` is already the ranking protocol (`bench.protocols[0]`) |
| M-02 no event volume | Partly mitigated: `tabular_v4_21` adds `volumen_local_integrado_m3`, `volumen_tributario_integrado_m3` and `integral_exceso_capacidad_m3`. XGBoost RMSE per shape 78.04 → 56.81 m³. NSE and per-shape error with v4_21 still to be measured |
| M-03 F1 averaged with zeros | Open (`ml/bench/metrics.py:41-43` and `mean_metrics`) |
| V-01 figures on a sample | Open (`visualization/resilience_curve.py`, `flood_volume_curve.py`; output of `--evaluate-shapes`) |
| V-07 `halve_timeseries.py` | Open |
| M-07/V-04 model ≠ `promotion.json` | In the worktree `outputs/models` is versioned and the SHA-256 values match `promotion.json`. An automatic verification and provenance in the figures are missing |
| D-04/H-06 label resolution | Open (`flood_threshold_m3: 1.0` = resolution of the `.rpt`) |
| OOF not persisted | Open (`main.py`, `_run_bench`: `_, metrics = evaluate_candidate(...)`) |

Test baseline before starting: **1,049 passed, 0 failed** (commit `1ecca50`).

## 2. Design decisions

### 2.1 Plug-in point
`score_predictions(prepared, oof, provenance)` in `ml/bench/evaluate.py` is the core layer: it scores out-of-fold predictions without knowing which model generated them. The no-learning references, the new M-03 aggregation and the per-shape breakdown enter there, so that all candidates and all references fall into the same table with the same metrics.

### 2.2 No-learning references (M-01)
New module `ml/bench/baselines.py`. Each reference receives a `PreparedDataset` and returns an OOF frame with the `OOF_COLUMNS` columns; it is then scored with `score_predictions`.
- `interpolacion_factor`: for a test row (shape, node, factor), it linearly interpolates the volume of the same node and shape between the neighbouring train factors. At the extremes it uses the nearest neighbour. It applies to protocols grouped by `factor_mult`. In `LeaveOneShapeOut` it does not apply (the shape is not in train) and is explicitly omitted.
- `forma_mas_cercana`: for a test row (held-out shape, node, factor), it takes the volume of the same node and factor in the nearest train shape in standardized (`duracion_horas`, `tiempo_al_pico_h`). It applies to `LeaveOneShapeOut`.
- Predicted label: `volumen_predicho >= threshold_m3`. Probability: equal to the predicted label (0/1). The ROC-AUC of a reference without real probabilities is not interpreted; this is documented.
- Each protocol carries in `schemas.py` a description of what it measures (`PROTOCOL_INTERPRETATION`): `LOSO` and `GroupKFold5` measure interpolation between factors; `LeaveOneShapeOut` measures generalization to new shapes. That description is written to the report JSONs.

### 2.3 Aggregation of classifier metrics (M-03)
For each protocol:
- `classifier` (main figure) = metrics computed over all the concatenated test rows of the protocol, plus `specificity`. The pooled confusion matrix `confusion_matrix = {tn, fp, fn, tp}` is a separate top-level key (not nested in `classifier`).
- `classifier_per_fold` = dispersion statistics across folds (`min`, `median`, `max`, `mean`) of precision, recall and F1, computed only over folds where the metric is defined. In a fold without true positives, recall and F1 are **NaN**, not 0. `n_undefined` is reported.
- `classifier_legacy_mean_zero_fill` = the old value (per-fold mean with `zero_division=0`), kept under an explicit name so that the before and after can be quoted.
- The `classifier.f1` tie-breaking criterion of the ranking now reads the pooled value.
- The parity test with `ml/evaluator.py` is adjusted to compare against `classifier_legacy_mean_zero_fill`: parity with the legacy code remains demonstrated, and the new value is tested separately.
- `regressor_oracle` and `end_to_end` do not change definition in this phase. A new block `end_to_end_pooled` is added (RMSE, MAE, true and predicted total volume and `vol_total_error_pct`, computed over the pool of all folds); `end_to_end` keeps exactly the keys of the legacy evaluator because the parity test requires it.

### 2.4 Per-shape breakdown (M-02)
`score_predictions` adds `by_shape` per protocol: for each `shape_id` in the test set, `n_rows`, `n_positives`, `f1`, `rmse_vol`, `vol_real_m3`, `vol_pred_m3` and `error_vol_pct`. It only makes sense in `LeaveOneShapeOut` (one shape per fold), but it is computed for any protocol.

### 2.5 Persistence of out-of-fold predictions (M-08/T-19)
`_run_bench` keeps the first element returned by `evaluate_candidate` and writes it to `outputs/bench/oof/<prep_id>/<familia>.parquet`. The provenance (`prep_id`, per-family hyperparameters, UTC date, evaluated and failed references) is written to a separate `provenance.json` in `outputs/bench/reports/`, not inside each metrics JSON; each `metrics_*.json` carries only two marker keys at the top level: `metrics_definition` (`classifier_pooled_v2`) and `prep_id`, to distinguish files from partial runs with different definitions. The references of 2.2 are persisted in the same way under the `baseline_*` family.

### 2.6 Figures and outputs on a sample (V-01)
The figures of `resilience_curve`, `flood_volume_curve` and the per-factor comparison add a footnote (the titles and legends do not change, on purpose, because other tests verify them): `"El modelo fue entrenado con estos escenarios; no es validación"` ("The model was trained on these scenarios; it is not validation"). `--evaluate-shapes` prints a warning in the header of its output. The out-of-sample versions derivable from the predictions of step 2.5 remain an optional improvement, outside this phase.

### 2.7 `halve_timeseries.py` (V-07)
It becomes a function `halve_timeseries(src, dst, force=False)` with `main()` under `if __name__ == "__main__"` and explicit `--src` and `--dst` arguments. It refuses to overwrite an existing `dst` without `force`. Importing the module has no side effects. The file is not deleted.

### 2.8 Model–`promotion.json` consistency (M-07/V-04)
- `verify_promotion(models_dir)` in `ml/bench/promote.py`: compares the SHA-256 of `classifier.joblib` and `regressor.joblib` with `promotion.json`. It returns a structured result (`ok`, list of discrepancies).
- It is invoked before loading the models of `outputs/models` to predict (`ml/predict.py`). On a discrepancy, it raises `ModelProvenanceError` with both SHA values. If `promotion.json` does not exist, it warns and continues (compatibility with legacy models).
- The figure-generation paths (`visualization/runner.py`) write `provenance.json` next to the figures: family, SHA-256 of the models, `prep_id`, date.
- Contract test: `--bench` and `--bench-evaluate` do not write to `outputs/models/`.
- `promote: "xgboost"` in `config.yaml` is not changed; promotion remains explicit with `--bench-promote`.

### 2.9 Label resolution (D-04/H-06) — spike before code
No code is changed in this step until a decision is made. The spike measures, on copies and in the scratchpad:
1. whether the flooded volume per node can be obtained from the binary `.out` (flooding series integrated over time) with a resolution better than 1 m³, and how much it differs from the `.rpt` in the control combinations;
2. the time per simulation and the estimated cost of re-simulating the 1,000 runs;
3. how many labels would change and what happens to the 0.5–2 m³ band when a continuous threshold is used.

With those numbers the user decides between:
- **A:** keep the current resolution and threshold, declare them as an assumption and report a confusion matrix that separates the borderline band (true volume from 1 to 2 m³);
- **B:** re-simulate, regenerate the dataset (new `prep_id`) and re-run the benchmark.

## 3. Files

| File | Change |
|---|---|
| `swmm_resilience/ml/bench/baselines.py` | New |
| `swmm_resilience/ml/bench/metrics.py` | Pooled aggregation, per-fold dispersion, confusion matrix |
| `swmm_resilience/ml/bench/evaluate.py` | `score_predictions` with the new metrics and `by_shape` |
| `swmm_resilience/ml/bench/schemas.py` | `PROTOCOL_INTERPRETATION` |
| `swmm_resilience/ml/bench/reports.py` | Separate `provenance.json`, `metrics_definition`/`prep_id` markers in each metrics JSON, protocol description |
| `swmm_resilience/ml/bench/ranking.py` / `config.yaml` | Path of the `classifier.f1` tie-breaker towards the pooled value |
| `swmm_resilience/ml/bench/promote.py`, `swmm_resilience/ml/predict.py` | `verify_promotion` and its use |
| `swmm_resilience/visualization/*.py`, `swmm_resilience/analysis/*.py`, `main.py` | In-sample labels, OOF, warning in `--evaluate-shapes`, `provenance.json` |
| `halve_timeseries.py` | Guard and function |
| `tests/ml/bench/…` | New tests and parity-test adjustment |
| `docs/auditoria/correcciones/` | Documentation (section 5) |

## 4. Tests

Each change starts from a failing test (TDD) and closes with the full suite green. Key tests:
- The aggregation: a perfect predictor gives pooled F1 = 1.0; in a fold without positives the fold recall is NaN and `n_undefined` counts it; the confusion matrix sums to the number of rows.
- The references: interpolation over a known linear function exactly reproduces the omitted values; the nearest shape picks the expected shape in a minimal case; the OOF frame satisfies `_validate_oof`.
- `by_shape`: relative error computed by hand in a 2-shape case.
- `verify_promotion`: matches, does not match and `promotion.json` missing.
- `halve_timeseries`: importing the module does not write; overwriting without `force` fails.
- Contract: `--bench` and `--bench-evaluate` do not touch `outputs/models`.
- Parity-test adjustment: the legacy value still matches `ml/evaluator.py`.

## 5. Documentation for the research

Directory `docs/auditoria/correcciones/`:
- `00-indice.md`: table of findings → status → commit → document.
- `01-evaluacion-honesta.md` (M-01, M-02, M-03, OOF and confusion matrix).
- `02-trazabilidad-y-seguridad.md` (V-01, V-07, M-07/V-04).
- `03-resolucion-de-la-etiqueta.md` (D-04/H-06; spike results and decision).
- `04-informe-consolidado.md`: before/after results, tables, and the summary that can be incorporated into the research report.

Each finding is documented with: problem and evidence, cause, change made (commit), verification with before and after figures, remaining limits and a **"Text for the thesis"** block.

## 6. Execution

- `subagent-driven-development`: one subagent per task, two-stage review; a powerful model (Opus) for metrics, references and measurements; Sonnet for mechanical tasks.
- One commit per task and push to `fix/cierre-auditoria-hidrogramas-20260927` at the close of each step.
- Gates: step 6 stops after the spike and waits for the A/B decision. Part 2 does not start until Part 1 is closed.

## 7. Out of scope

- Redesigning the event-volume features beyond what v4_21 provides (decided after measuring in step 1).
- Out-of-sample resilience curves (optional later improvement).
- SVM, fair tuning and final selection: Part 2.
- Aligning the main checkout with the remote branch.
