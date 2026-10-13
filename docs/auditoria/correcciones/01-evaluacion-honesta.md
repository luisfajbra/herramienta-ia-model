# Correction 01 — Honest evaluation of the surrogate model (M-01, M-02, M-03, M-08)

- **Date:** 2026-10-11
- **Branch:** `worktree-optuna-tuning-familia-ganadora` (commits `b831f0f` … `284927d`; full list in [`04-informe-consolidado.md`](04-informe-consolidado.md)).
- **Origin of the findings:** [`../2026-10-10-auditoria-completa.md`](../2026-10-10-auditoria-completa.md) and [`../anexos/02-ml-protocolo-evaluacion.md`](../anexos/02-ml-protocolo-evaluacion.md).
- **Source of all new figures:** [`datos/`](datos/README.md). Each number carries its file and its column.

## 0. Scope of the measurements (read before quoting any figure)

| Aspect | Value | Source |
|---|---|---|
| Evaluated model | XGBoost with the **default hyperparameters of `config.yaml`** (200 trees, depth 6, learning rate 0.05, subsampling 0.8, in classifier and regressor). It is **not** the model tuned with Optuna nor the promoted model | `datos/runs/provenance.json`, `xgboost_params` and `xgboost_params_note` |
| Feature contract | `tabular_v4_21` (21 features) | `datos/tables_provenance.json`, `feature_contract_id`; `datos/runs/provenance.json`, `n_features` |
| Prepared dataset | `prep_id` `ca706e27654e0cf5` | `datos/tables_provenance.json`, `prep_id` |
| Rows | 154,080 = 963 runs (shape, factor) × 160 nodes | `datos/tables_provenance.json`, `n_rows`, `n_runs` |
| Excluded runs | 37 of 1000 (continuity filter, \|error\| ≤ 5 %) | `datos/tables_provenance.json`, `n_runs_full_grid`, `n_missing_cells`, `max_continuity_error_pct` |
| Network | A single network (`chico_hydro-qx1`), 160 nodes | `datos/README.md`; annex 02 §1.3 |
| Label threshold | `inunda = 1` if the `.rpt` volume is ≥ 1 m³ (equivalent to ≈ 0.5 m³ of continuous volume in the re-simulated sample; see [`03`](03-resolucion-de-la-etiqueta.md)) | `datos/tables_provenance.json`, `flood_threshold_m3` |
| No-learning references | `baseline_interpolacion_factor` (LOSO, GroupKFold5) and `baseline_forma_mas_cercana` (LeaveOneShapeOut) | `datos/runs/provenance.json`, `baselines` |
| Fallback rows of the references | Interpolation: 3,840 of 308,160 rows (1.25 %), adding LOSO and GroupKFold5. Nearest shape: 640 of 154,080 (0.42 %) | `datos/runs/provenance.json`, `baseline_fallback` |

**Previous figures.** The "before" figures in this document come from the audit (annex 02), which was measured on **another dataset**: `tabular_v3_17` contract (17 features), 160,000 rows without continuity filter, PreparedDataset `af3751551ca5df98`. The contract and the set of rows change at the same time, so the before/after comparison **is not a controlled experiment**: it describes two states of the project, not the isolated effect of one change. The only controlled comparison of contracts available is that of [`../../resultados/2026-10-06-explicacion-completa.md`](../../resultados/2026-10-06-explicacion-completa.md) §11.1 (per-shape mean RMSE 78.0368 → 56.8088 m³), which was not re-run here; the run of this series reproduces the value 56.8088 (`datos/comparacion_protocolos.csv`, `rmse_vol_media_por_fold_m3` (per-fold-mean volume RMSE), XGBoost, LeaveOneShapeOut).

**What each protocol measures** (column `que_mide` (what it measures) of `datos/comparacion_protocolos.csv`):

| Protocol | Held-out group | What it measures |
|---|---|---|
| LOSO | One complete intensity factor | Interpolation between factors: all shapes remain in training |
| GroupKFold5 | Five factors per fold | Same as LOSO, with five factors held out per fold |
| LeaveOneShapeOut | One complete hydrograph shape | Generalization to unseen hydrograph shapes. **Main protocol** |

No protocol holds out nodes or the "new shape × new factor" combination. Every generalization claim is bounded to this network and these 160 nodes (annex 02 §1.3).

**Vocabulary.** "Pooled" = metric computed over all the out-of-fold predictions of the protocol, concatenated. "Per-fold mean" = average of the metric of each fold (the previous definition of the benchmark). "Oracle" = regressor evaluated only on the nodes that actually flooded. "End-to-end" = volume predicted by the regressor only where the classifier predicts flooding, evaluated on all nodes.

---

## 1. M-01 — LOSO and GroupKFold5 measure interpolation; no-learning references were missing

### Problem and evidence

The audit showed that, in LOSO, a lookup table that interpolates the volume of the same node and the same shape between neighbouring factors outperforms XGBoost (annex 02, M-01, `tabular_v3_17` dataset of 160,000 rows): pooled end-to-end RMSE of 9.00 m³ versus 68.43 m³, and pooled F1 of 0.9718 versus 0.9656. The benchmark had no no-learning reference to compare against, so a good number in LOSO did not allow knowing whether the model contributed anything.

### Cause

LOSO and GroupKFold5 group by `factor_mult` (annex 02 §1.3). The dynamic features are exactly `base × factor`, and when a factor is held out the same node with the same shape is in training at ±0.2. With a dense grid of 25 factors, predicting the held-out factor is an almost linear interpolation that does not require learning hydraulics.

### What changed

| Commit | File | Change |
|---|---|---|
| `9c8db1b` | `swmm_resilience/ml/bench/baselines.py` (new), `swmm_resilience/ml/bench/schemas.py` | Two no-learning references that produce out-of-fold predictions in the same format as a model and are scored with the same function (`score_predictions`). `PROTOCOL_INTERPRETATION` describes what each protocol measures and is written to the reports |
| `02cdaab`, `6c48c46` | `baselines.py` | The nearest-shape reference accepts an explicit shape descriptor table (the `tabular_v4_21` contract contains neither `duracion_horas` nor `tiempo_al_pico_h`) and rejects tables with NaN, duplicate shapes or missing columns |
| `833760d`, `5f0586d` | `baselines.py` | The references work on the incomplete grid left by the continuity filter (37 missing shape–factor cells) and count the rows that use a fallback |
| `a26155e`, `4a2a87c`, `4717885`, `e26fc5a` | `scripts/ejecutar_evaluacion_v4_21.py`, `scripts/informe_evaluacion.py` | Run of XGBoost and of the two references with the three protocols on `tabular_v4_21`, and report tables with their provenance |
| `395a2ee`, `284927d` | `docs/auditoria/correcciones/datos/` | Tables, full metrics, provenance and README |

Definition of the references (docstring of `baselines.py`; `datos/README.md`):

- **Factor interpolation** (LOSO, GroupKFold5): for each test row (shape, node, factor), it linearly interpolates the volume of the same node and shape between the neighbouring training factors available in that series; at the extremes it uses the available neighbour. The predicted label is `volume ≥ 1 m³`.
- **Nearest shape** (LeaveOneShapeOut): copies the volume of the same node and factor from the training shape most similar in (`duracion_horas`, `tiempo_al_pico_h`) standardized with the training shapes; if that shape does not have the cell, it falls to the next in the ordering. The descriptors are taken from the prepared dataset `tabular_v3_17` `2db0578f875526cb`, which has the same 40 shapes (`datos/runs/provenance.json`, `shape_descriptors_source`).
- The references do not produce probabilities: their "probability" is the label itself (0/1), so their ROC-AUC is not interpreted (spec §2.2).

### Verification

**Table 1.1 — Before (annex 02, M-01; `tabular_v3_17`, 160,000 rows).**

| Protocol | Model | Pooled F1 | Pooled end-to-end RMSE (m³) | Per-shape mean RMSE (m³) |
|---|---|---|---|---|
| LOSO | Factor interpolation | 0.9718 | 9.00 | — |
| LOSO | XGBoost | 0.9656 | 68.43 | — |
| LeaveOneShapeOut | Nearest shape | 0.9758 | 170.2 | 98.2 |
| LeaveOneShapeOut | XGBoost | 0.9789 | 136.1 | 79.7 |

**Table 1.2 — After (`tabular_v4_21`, 154,080 rows).** Source: `datos/comparacion_protocolos.csv`, columns `f1_agregado` (pooled F1), `recall_agregado` (pooled recall), `precision_agregada` (pooled precision), `rmse_vol_agregado_m3` (pooled volume RMSE), `rmse_vol_media_por_fold_m3`, `error_vol_total_pct` (total volume error), `nse_regresor_oracle` (oracle regressor NSE).

| Protocol | Model | Pooled F1 | Recall | Precision | Pooled RMSE (m³) | Per-fold mean RMSE (m³) | Total vol. error (%) | Oracle NSE |
|---|---|---|---|---|---|---|---|---|
| LOSO | Factor interpolation | 0.9718 | 0.9910 | 0.9534 | **9.13** | 4.77 | −0.39 | 0.9992 |
| LOSO | XGBoost | 0.9667 | 0.9810 | 0.9528 | **57.89** | 46.28 | −5.03 | 0.9675 |
| GroupKFold5 | Factor interpolation | 0.9718 | 0.9910 | 0.9534 | 9.13 | 7.47 | −0.39 | 0.9992 |
| GroupKFold5 | XGBoost | 0.9681 | 0.9817 | 0.9549 | 56.19 | 55.44 | −4.64 | 0.9694 |
| LeaveOneShapeOut | Nearest shape | 0.9764 | 0.9808 | 0.9721 | **172.86** | 98.14 | −2.98 | 0.7102 |
| LeaveOneShapeOut | XGBoost | **0.9817** | 0.9918 | 0.9718 | **106.21** | 56.81 | −7.98 | 0.8906 |

**Reading.**

1. **In LOSO, the no-learning reference beats XGBoost in volume** (pooled RMSE 9.13 versus 57.89 m³) and in F1 (0.9718 versus 0.9667). Finding M-01 holds with the `tabular_v4_21` contract: LOSO and GroupKFold5 measure interpolation between factors, and in that task a lookup table is better than the model.
2. **GroupKFold5 is not more demanding than LOSO for interpolation.** The predictions of the interpolation reference are identical in the two protocols (maximum absolute difference 0.0 over the 154,080 rows): with 25 factors distributed in an alternating way over 5 folds, no fold holds out two contiguous factors (fold 0: {0.6, 2.0, 3.0, 4.0, 5.0}; fold 1: {0.4, 1.8, 2.8, 3.8, 4.8}; …), so each held-out factor keeps its two neighbours in training. Source: the implementer's check in the report of task 8 (unversioned working notes), repeated for this document with the command of §5. For the same reason, in Table 1.2 the interpolation rows coincide between LOSO and GroupKFold5 in all the pooled metrics; only the per-fold mean changes, because the folds group different rows.
3. **LeaveOneShapeOut is the protocol that measures something that interpolation does not solve.** There XGBoost beats the nearest-shape reference in volume (pooled RMSE 106.21 versus 172.86 m³; per-fold mean 56.81 versus 98.14 m³; oracle NSE 0.8906 versus 0.7102) and, by a small margin, in F1 (0.9817 versus 0.9764). The reference has, however, a smaller total volume error (−2.98 % versus −7.98 %). Per shape (derived from `datos/por_forma.csv`, comparing `rmse_vol` and `error_vol_pct` of both models in LeaveOneShapeOut), XGBoost has a lower RMSE in 31 of the 40 shapes, a lower absolute total volume error in 33 and a higher F1 in 25.
4. **The fallback rows are few.** The interpolation uses a neighbour other than the immediate one of the fold grid in 1.25 % of its rows, and the nearest shape falls to the second or following shape in the ordering in 0.42 % (`datos/runs/provenance.json`, `baseline_fallback`). The interpolation counter is defined against the training grid of the fold, not against the full grid, so as not to count as fallback the factors that GroupKFold5 holds out on purpose (docstring of `baselines.py`).

**Tests supporting the change** (`tests/ml/bench/test_baselines.py`, 30 tests): among others, `test_interpolation_reproduces_a_linear_function_in_the_interior`, `test_nearest_shape_borrows_the_volume_of_the_closest_training_shape`, `test_interpolation_uses_the_series_own_neighbours_when_a_middle_factor_is_missing` (non-linear series, values computed by hand), `test_nearest_shape_walks_down_the_ranking_when_the_nearest_lacks_the_cell`, `test_fallback_counters_for_the_interpolation_baseline` and `test_every_protocol_has_an_interpretation`.

### Remaining limits

- **Untuned XGBoost.** The XGBoost figures are those of the default hyperparameters of `config.yaml`; the model tuned with Optuna was not evaluated here against the references.
- **A single network and 963 of 1000 runs.** No protocol holds out nodes or networks; generalization to another network is not measured.
- **The nearest-shape reference depends on how "nearest" is defined.** It uses only duration and time to peak. With another definition (for example, including the hydrograph volume) its number would change. An example of its fragility (`datos/por_forma.csv`): `shape_quasi_uniform_block` and `shape_slow_broad_rise` choose each other as neighbours and the reference swaps their volumes (1,767,048 and 684,492 m³), with a total volume error of −61.3 % and +158.2 %.
- **Fallback rows.** 1.25 % and 0.42 % of the rows of the references do not use their primary source because the continuity filter left gaps in the grid.
- **There is no extrapolation protocol.** Annex 02 recommended a combined protocol (new shape and factor at the same time, or blocks of extreme factors). It was not implemented.
- **The benchmark ranking did not change its main metric.** It still orders by the per-fold mean of the end-to-end RMSE, without confidence intervals (M-04, open).

### Text for the thesis

> To interpret the cross-validation metrics, the models were compared with two references that do not learn. The first estimates the volume of a node by linearly interpolating, for the same hydrograph shape, the volumes simulated at the neighbouring intensity factors; the second copies the volume of the same node and factor from the training shape most similar in duration and time to peak. In the protocols that hold out intensity factors (LOSO and GroupKFold5), interpolation obtains a volume RMSE of 9.13 m³ versus 57.89 m³ for XGBoost in LOSO, and an F1 of 0.9718 versus 0.9667. These protocols therefore measure the ability to interpolate between already seen factors with the same shape, and provide no evidence of generalization; with the 25-factor grid used, GroupKFold5 is not more demanding than LOSO, because no fold holds out two contiguous factors. The main protocol of this work is validation leaving out one complete hydrograph shape (LeaveOneShapeOut). In it, XGBoost beats the nearest-shape reference in volume (pooled RMSE of 106.21 versus 172.86 m³) and, by a small margin, in classification (F1 of 0.9817 versus 0.9764). These figures correspond to XGBoost with default hyperparameters, to a single drainage network and to 963 of the 1000 simulations, those that meet a continuity error of at most 5 %.

---

## 2. M-02 — Volume generalizes worse to unseen shapes

### Problem and evidence

With the previous contract (`tabular_v3_17`), the audit measured in LeaveOneShapeOut an oracle regressor NSE of 0.8141, versus 0.9530 in LOSO, and a total volume error per shape from −63.3 % (`shape_quasi_uniform_block`) to +216.3 % (`common_bogota`). A single shape, `shape_quasi_uniform_block`, concentrated 61.35 % of the end-to-end squared error (annex 02, M-02; audit §3.1). The benchmark had no per-shape breakdown, so these figures did not come from any report of the project.

### Cause

In `tabular_v3_17` the hydrograph shape enters the model only as `duracion_horas` and `tiempo_al_pico_h`, and the flows are normalized by the peak, so that no feature describes the event volume (annex 02, M-02; annex 01, H-02). The `tabular_v4_21` contract, prior to this series of corrections, adds integrated volume variables (`volumen_local_integrado_m3`, `volumen_tributario_integrado_m3`, `integral_exceso_capacidad_m3`, among others; `datos/runs/provenance.json`, `feature_columns`). What was missing was measuring the NSE and the per-shape error with it (spec §1).

### What changed

This series changes neither the model nor the features. It adds the **measurement**:

| Commit | File | Change |
|---|---|---|
| `b831f0f` | `swmm_resilience/ml/bench/metrics.py` | `shape_breakdown_entry` (F1, RMSE and total volume error per shape) and `end_to_end_pooled_metrics` (pooled RMSE, MAE and total volume error) |
| `bd0f23f` | `swmm_resilience/ml/bench/evaluate.py` | `score_predictions` returns `by_shape` and `end_to_end_pooled` for each protocol |
| `a26155e` … `284927d` | `scripts/`, `docs/auditoria/correcciones/datos/` | Run on `tabular_v4_21` and table `por_forma.csv` (6 model–protocol combinations × 40 shapes) |

### Verification

**Table 2.1 — Before and after.** "Before": annex 02 (`tabular_v3_17`, 160,000 rows). "After": `tabular_v4_21`, 154,080 rows. It is not a controlled comparison (§0).

| Metric (XGBoost, LeaveOneShapeOut unless indicated) | Before | "Before" source | After | "After" source |
|---|---|---|---|---|
| Oracle regressor NSE | 0.8141 | Audit §3.1 | 0.8906 | `comparacion_protocolos.csv`, `nse_regresor_oracle` |
| Oracle NSE in LOSO | 0.9530 | Audit §3.1 | 0.9675 | same |
| Oracle NSE in GroupKFold5 | 0.9553 | Audit §3.1 | 0.9694 | same |
| Oracle regressor RMSE (m³) | 253 | Annex 02, M-02 | 193.91 | `runs/metrics_xgboost.json`, `LeaveOneShapeOut.regressor_oracle.rmse` |
| Pooled end-to-end RMSE (m³) | 136.1 | Annex 02, M-04 | 106.21 | `comparacion_protocolos.csv`, `rmse_vol_agregado_m3` |
| Per-shape mean end-to-end RMSE (m³) | 79.7 | Annex 02, M-04 | 56.81 | `comparacion_protocolos.csv`, `rmse_vol_media_por_fold_m3` |
| Total volume error per shape, minimum | −63.3 % (`shape_quasi_uniform_block`) | Annex 02, M-02 | −52.1 % (`shape_quasi_uniform_block`) | `por_forma.csv`, `error_vol_pct` |
| Total volume error per shape, maximum | +216.3 % (`common_bogota`) | Annex 02, M-02 | +33.3 % (`shape_four_pulse_cluster`) | `por_forma.csv`, `error_vol_pct` |
| Share of the squared error of the worst shape | 61.35 % (`shape_quasi_uniform_block`) | Annex 02, M-02 | 71.6 % (`shape_quasi_uniform_block`) | Derived from `por_forma.csv`: `rmse_vol² × n_rows` of the shape / sum of the 40 shapes |
| Total volume error, all shapes | — | — | −7.98 % | `comparacion_protocolos.csv`, `error_vol_total_pct` |

Figures derived from `datos/por_forma.csv` (XGBoost, LeaveOneShapeOut), computed for this document with the command of §5:

- Shapes with \|total volume error\| > 20 %: 12 of 40. Median of the \|error\| per shape: 18.2 %.
- The two shapes with the largest squared error add up to 77.6 % (`shape_quasi_uniform_block` 71.6 % and `shape_sustained_plateau_14h` 5.9 %); the first three, 81.5 %.
- Without `shape_quasi_uniform_block`, the pooled RMSE of the other 39 shapes would be 57.32 m³ (versus 106.21 m³ with the 40).
- The worst-classified shape is `shape_ultra_short_impulse` (F1 0.8798), followed by `common_bogota` (0.9188) and `rare_extreme` (0.9316); these three are also the ones that lost the most cells to the continuity filter (`n_rows` 2,400, 3,200 and 2,880 of 4,000).

The full per-shape table is in §6.

**Reading.**

- With `tabular_v4_21`, the oracle NSE in LeaveOneShapeOut rises from 0.8141 to 0.8906 and the range of the per-shape error narrows from [−63.3 %, +216.3 %] to [−52.1 %, +33.3 %]. Attributing that improvement to the volume variables is an **interpretation**: the two measurements also differ in the set of rows. The controlled comparison of contracts (same set of rows) exists only for the per-shape mean RMSE: 78.0368 → 56.8088 m³ (`docs/resultados/2026-10-06-explicacion-completa.md` §11.1, not re-run here).
- **The concentration of the error did not go down: it went up.** The same shape (`shape_quasi_uniform_block`, underestimated by 52.1 %) goes from 61.35 % to 71.6 % of the squared error. That this happens because the other shapes improved more is an **interpretation** compatible with the data (the other 39 shapes give a pooled RMSE of 57.32 m³), not a measurement of the mechanism.
- **The pooled RMSE and the per-shape mean tell different stories.** In LeaveOneShapeOut, 106.21 versus 56.81 m³. The per-fold mean dilutes the heavy-tail shapes and the pooled value weights them by their squared error. Both must be reported, stating which one answers which question (M-04).

### Remaining limits

- **M-02 remains open for block-shaped hydrographs.** The volume of `shape_quasi_uniform_block` is underestimated by 52.1 % and that shape dominates the pooled error.
- **The reason was not measured.** The correlation of the per-shape error with the filling factor (Spearman −0.577 in `tabular_v3_17`, annex 02) was not recomputed with `tabular_v4_21`.
- **No ablation of the volume features was done under LeaveOneShapeOut** (recommendation of annex 02). The improvement attributable to each variable is not measured.
- **Redesigning the volume features beyond `tabular_v4_21`** was left out of scope (spec §7).
- **LeaveOneShapeOut is the optimistic extreme of per-shape generalization**: it holds out one shape and leaves 39 in training. With 8 shapes held out per fold, the audit measured in `tabular_v3_17` an oracle NSE of 0.69 (annex 02, M-02); it was not measured with `tabular_v4_21`.
- **Untuned XGBoost, one network, 963 runs** (§0).

### Text for the thesis

> Generalization to unseen hydrograph shapes was evaluated by leaving out, in each fold, one of the 40 shapes (LeaveOneShapeOut). With the 21-variable contract, which includes integrated variables of the event volume, the regressor evaluated on the nodes that actually flooded reaches an NSE of 0.8906, versus 0.9675 when intensity factors are held out. The error of the predicted total volume per shape ranges from −52.1 % to +33.3 %, and a single shape, with an almost uniform block inflow, concentrates 71.6 % of the squared error: its volume is underestimated by 52.1 %. Without it, the pooled RMSE of the other 39 shapes would be 57.32 m³, versus 106.21 m³ with all of them. Classification generalizes better than the magnitude of the volume: the pooled F1 is 0.9817, with the worst per-shape value (0.8798) at the ultra-short impulse, which is also the shape with the most simulations excluded by the continuity criterion. With the previous 17-variable contract, the project audit had measured an NSE of 0.8141 and a per-shape error range of −63.3 % to +216.3 %; that measurement used a different set of rows, so the difference is not attributed entirely to the change of variables. These figures correspond to XGBoost with default hyperparameters and to a single network.

---

## 3. M-03 — F1, precision and recall averaged per fold with zeros

### Problem and evidence

The benchmark averaged F1, precision and recall per fold with `zero_division=0`, so that a fold with no real flooding scored 0 even if the classifier got everything right. In LOSO there are 5 such folds (low factors without flooded nodes). The audit measured, with `tabular_v3_17` and 160,000 rows, a published F1 of 0.7408 in LOSO versus a pooled F1 of 0.9656, and a precision of 0.7226 versus 0.9508. A **perfect** predictor obtained F1 = 0.80 in LOSO (annex 02, M-03; audit §3.1). The "gap" between LOSO and GroupKFold5 in the reports was an artefact of that aggregation.

### Cause

`ml/bench/metrics.py:41-43` (`zero_division=0`) and the per-fold mean in `evaluate.py` and in the legacy evaluator `ml/evaluator.py` (annex 02, M-03).

### What changed

| Commit | File | Change |
|---|---|---|
| `b831f0f` | `swmm_resilience/ml/bench/metrics.py` | Pure functions: `confusion_counts`, `pooled_classifier_metrics` (over all the concatenated test rows, with specificity), `fold_classifier_metrics` (NaN where the metric is not defined) and `summarize_fold_metrics` (minimum, median, maximum, mean and number of undefined folds) |
| `2871917` | same | The pooled metrics are also NaN, not 0, when they are not defined |
| `bd0f23f` | `swmm_resilience/ml/bench/evaluate.py`, `tests/ml/bench/test_parity_with_legacy_evaluator.py` | `classifier` becomes the pooled value; `classifier_per_fold` gives the dispersion across folds; the previous value is kept as `classifier_legacy_mean_zero_fill`. The parity test with the legacy evaluator now compares against that field |
| `5528479` | `tests/ml/bench/test_score_predictions.py`, spec | The `protocols` parameter is tested with a set of two protocols; the spec is aligned with the code names (`n_undefined`) |

Since the ranking tie-breaking criterion is `classifier.f1` (`config.yaml`, `bench.ranking.tie_breakers`), that tie-breaker now reads the pooled value.

### Verification

**Table 3.1 — F1 in LOSO, previous definition versus new.**

| Case | Zero-filled per-fold mean (previous) | Pooled (new) | Folds with undefined F1 | Source |
|---|---|---|---|---|
| XGBoost, `tabular_v3_17`, 160,000 rows (before) | 0.7408 | 0.9656 | 5 | Annex 02, M-03 |
| XGBoost, `tabular_v4_21` (after) | 0.7379 | 0.9667 | 5 | `comparacion_protocolos.csv`, `f1_media_por_fold_con_ceros` (zero-filled per-fold mean F1), `f1_agregado`, `folds_f1_indefinido` (folds with undefined F1) |
| Factor interpolation, `tabular_v4_21` | 0.7425 | 0.9718 | 5 | same |
| Perfect predictor, `tabular_v3_17` (before) | 0.80 | — | 5 | Annex 02, M-03 (script `m_08`) |
| Perfect predictor, `tabular_v4_21` (after) | 0.8000 | 1.0000 | 5 | Command of §5 (run for this document) |

**Table 3.2 — XGBoost, `tabular_v4_21`, all the classifier metrics in LOSO.** Source: `datos/runs/metrics_xgboost.json`, block `LOSO`.

| Metric | Zero-filled per-fold mean (`classifier_legacy_mean_zero_fill`) | Pooled (`classifier`) | Per defined fold: minimum / median / maximum (`classifier_per_fold`) |
|---|---|---|---|
| Precision | 0.7215 | 0.9528 | 0.2541 / 0.9488 / 0.9959 |
| Recall | 0.7601 | 0.9810 | 0.5679 / 0.9807 / 0.9972 |
| F1 | 0.7379 | 0.9667 | 0.3511 / 0.9648 / 0.9931 |
| Specificity | — | 0.9792 | — |

**Table 3.3 — The "gap" between protocols disappears.** Source: `datos/comparacion_protocolos.csv` (XGBoost).

| Protocol | Zero-filled per-fold mean F1 | Pooled F1 |
|---|---|---|
| LOSO | 0.7379 | 0.9667 |
| GroupKFold5 | 0.9672 | 0.9681 |
| LeaveOneShapeOut | 0.9809 | 0.9817 |

**Reading.** With the previous definition, LOSO looked 0.23 points of F1 worse than GroupKFold5 (0.7379 versus 0.9672); with the pooled one, the difference is 0.0014 (0.9667 versus 0.9681). In GroupKFold5 and LeaveOneShapeOut there are no undefined folds and the two definitions almost coincide. The per-fold dispersion in LOSO is large (F1 from 0.3511 to 0.9931 across the folds with flooding): the minimum corresponds to factor 1.2 (`runs/metrics_xgboost.json`, `LOSO.by_factor["1.20"].f1` = 0.3511), near the onset of flooding.

**Tests** (`tests/ml/bench/test_score_predictions.py`, `test_metrics.py`): `test_perfect_predictions_score_f1_one_even_with_a_fold_without_flooding`, `test_per_fold_dispersion_reports_the_undefined_fold`, `test_the_legacy_zero_fill_mean_is_kept_under_an_explicit_name` and, in the parity test, `test_the_new_classifier_value_is_pooled_and_differs_from_the_legacy_mean`.

### Remaining limits

- **`by_factor` did not change.** It still averages with zeros: in XGBoost LOSO, factors 0.20 and 1.00 give F1 = 0.0 (`runs/metrics_xgboost.json`, `LOSO.by_factor`). It must not be quoted for factors without flooding.
- **The legacy evaluator did not change.** `ml/evaluator.py` and the JSONs in `outputs/metrics/` (for example `metrics_classifier.json`, with F1 0.7408) keep the previous definition. Those figures must not be quoted.
- **No confidence intervals.** The per-fold dispersion is reported, but not a CI by group resampling (M-04, M-11, open).
- **Implicit 0.5 threshold** in the classifier (M-09, open).

### Text for the thesis

> The classification metrics are computed on the pooled confusion matrix of all the out-of-fold predictions of each protocol, and the variation across folds is reported separately (minimum, median and maximum), excluding the folds where the metric is not defined. This choice matters in the validation that holds out intensity factors: five of the 25 factors produce no flooded node and, if F1 = 0 is assigned in those folds and then averaged, XGBoost obtains an F1 of 0.7379 instead of the pooled value of 0.9667, and a perfect predictor would obtain 0.80. The apparent difference between LOSO and GroupKFold5 in earlier versions of this work was an effect of that aggregation: with the pooled matrix, the F1 values are 0.9667 and 0.9681.

---

## 4. M-08, M-10 and T-19 — Persisted out-of-fold predictions and confusion matrix

### Problem and evidence

`evaluate_candidate()` produces one prediction per row, but `main.py` discarded it (`_, metrics = evaluate_candidate(...)`), and the reports saved only per-fold averages, without `prep_id` or hyperparameters. A confusion matrix could not be reconstructed, nor could one know which dataset each report came from (audit §6.1; annex 02, M-08 and M-10). The only persisted matrix was that of the SQL table `oof_predictions`, only for XGBoost with GroupKFold5.

### Cause

The benchmark was designed to rank with per-fold averages; per-row persistence and the confusion matrix were not part of the report (annex 02 §4.1).

### What changed

| Commit | File | Change |
|---|---|---|
| `bd0f23f` | `swmm_resilience/ml/bench/evaluate.py` | Key `confusion_matrix` = {tn, fp, fn, tp} pooled per protocol; specificity in `classifier`; `by_shape` and `end_to_end_pooled` (pooled RMSE, MAE and total volume error) |
| `b95a480` | `main.py`, `swmm_resilience/ml/bench/reports.py` | `--bench-evaluate` saves the out-of-fold predictions of each family and each reference to `outputs/bench/oof/<prep_id>/<nombre>.parquet`, and a `provenance.json` with `prep_id`, hyperparameters, references, date and what each protocol measures |
| `4a2a87c` | `scripts/ejecutar_evaluacion_v4_21.py` | The `tabular_v4_21` run persists predictions and metrics when each stage finishes |
| `e26fc5a`, `284927d` | `scripts/informe_evaluacion.py`, `datos/tables_provenance.json`, `datos/README.md` | The tables record the command, the commit, the SHA-256 of each predictions file read and of each CSV written |

### Verification

**Table 4.1 — Pooled confusion matrices, `tabular_v4_21`.** Source: `datos/matriz_confusion.csv` (TN, FP, FN, TP); specificity from `datos/runs/metrics_*.json`, `classifier.specificity`. In each row TN + FP + FN + TP = 154,080.

| Model | Protocol | TN | FP | FN | TP | Specificity |
|---|---|---|---|---|---|---|
| XGBoost | LeaveOneShapeOut | 106,526 | 1,329 | 380 | 45,845 | 0.9877 |
| Nearest shape | LeaveOneShapeOut | 106,556 | 1,299 | 889 | 45,336 | 0.9880 |
| XGBoost | LOSO | 105,610 | 2,245 | 879 | 45,346 | 0.9792 |
| Factor interpolation | LOSO | 105,617 | 2,238 | 416 | 45,809 | 0.9792 |
| XGBoost | GroupKFold5 | 105,711 | 2,144 | 844 | 45,381 | 0.9801 |
| Factor interpolation | GroupKFold5 | 105,617 | 2,238 | 416 | 45,809 | 0.9792 |

**Before.** The audit regenerated the predictions in a temporary folder (not persisted) and obtained, for XGBoost in LeaveOneShapeOut with `tabular_v3_17` and 160,000 rows: TN 112,153, FP 1,605, FN 373, TP 45,869 (audit §6.2). The totals differ because the set of rows is another; the two matrices are not compared cell by cell.

**Table 4.2 — False negatives by true volume.** Source: `datos/banda_limitrofe.csv`. The band [1, 2) m³ contains only the integer 1 (continuous volume of ≈ 0.5 to 1.5 m³; see [`03`](03-resolucion-de-la-etiqueta.md) §6.1).

| Model | Protocol | Total FN | FN with true volume in [1, 2) m³ | FN outside the band | FN with true volume ≥ 100 m³ |
|---|---|---|---|---|---|
| XGBoost | LeaveOneShapeOut | 380 | 333 | 47 | 0 |
| Nearest shape | LeaveOneShapeOut | 889 | 611 | 278 | 0 |
| XGBoost | LOSO | 879 | 443 | 436 | 4 |
| Factor interpolation | LOSO | 416 | 393 | 23 | 0 |
| XGBoost | GroupKFold5 | 844 | 435 | 409 | 5 |
| Factor interpolation | GroupKFold5 | 416 | 393 | 23 | 0 |

In LeaveOneShapeOut, 333 of the 380 false negatives of XGBoost are in the 1 m³ band and none has a true volume ≥ 100 m³. The full matrix by volume class and the metrics with and without the band are in [`03`](03-resolucion-de-la-etiqueta.md) §6.1–6.2 (F1 0.9817 with all rows; 0.9847 without the band, as a sensitivity analysis).

**Traceability.** `datos/tables_provenance.json` records the SHA-256 of the three predictions files (`inputs_sha256`) and of the four CSVs (`outputs_sha256`), the command and the commit `e26fc5a` with a clean versioned tree (`git_dirty: false`). Tests: `test_confusion_matrix_counts_every_test_row`, `test_a_false_negative_and_a_false_positive_show_up_in_the_confusion_matrix`, `test_bench_evaluate_persists_oof_baselines_and_provenance`, `test_write_provenance_records_prep_id_params_baselines_and_protocol_meaning`.

### Remaining limits

- **The predictions files are not versioned** (`outputs/` is in `.gitignore`). They are reproduced with the commands of `datos/README.md` (≈ 6 min) and verified with the SHA-256 values of `tables_provenance.json`.
- **The `metrics_<familia>.json` JSONs of `--bench-evaluate` do not carry the `prep_id` in the name**: a file from an earlier run can sit next to a newer `provenance.json` (deferred; see `04`).
- **Metrics still missing** (M-10 partial): MCC, balanced accuracy and PR-AUC.
- **Implicit 0.5 decision threshold** and uncalibrated probabilities (M-09, open).
- **The reading of the band FNs** as difficulty inherent to the boundary is an interpretation (see `03` §6.3).

### Text for the thesis

> The out-of-fold predictions of each model and each reference are saved per row, together with the identifier of the prepared dataset, the hyperparameters and the SHA-256 fingerprint of each file, so that all the evaluation tables can be reconstructed without retraining. With them the pooled confusion matrix is obtained. For XGBoost in the held-out-shape validation, over 154,080 node-scenario cases, there are 45,845 true positives, 106,526 true negatives, 1,329 false positives and 380 false negatives (recall of 0.9918 and specificity of 0.9877). Of those 380 false negatives, 333 correspond to nodes whose reported volume is 1 m³, the minimum that the label considers flooding, and none to nodes with 100 m³ or more.

---

## 5. Commands used for the figures derived in this document

All of them read existing files and do not write to the repository. They are run from the root of the worktree with `../../../venv/Scripts/python.exe`.

**Per-shape table, share of the squared error and comparison with the reference** (reads `datos/por_forma.csv`):

```python
import pandas as pd, numpy as np
d = pd.read_csv("docs/auditoria/correcciones/datos/por_forma.csv")
x = d[(d.modelo == "xgboost") & (d.protocolo == "LeaveOneShapeOut")].set_index("forma")
b = d[d.modelo == "baseline_forma_mas_cercana"].set_index("forma")
x["sse"] = x.rmse_vol ** 2 * x.n_rows
share = 100 * x.sse / x.sse.sum()                      # % del error cuadrático por forma
w = x.sse.idxmax()                                     # shape_quasi_uniform_block, 71,6 %
rmse_sin_peor = np.sqrt((x.sse.sum() - x.sse[w]) / (x.n_rows.sum() - x.n_rows[w]))   # 57,32
print(w, share[w], rmse_sin_peor, share.sort_values(ascending=False).head(3).cumsum().tolist())
print((x.error_vol_pct.abs() > 20).sum(), x.error_vol_pct.abs().median())              # 12; 18,2
print((x.rmse_vol < b.rmse_vol.reindex(x.index)).sum(),                                 # 31
      (x.error_vol_pct.abs() < b.error_vol_pct.reindex(x.index).abs()).sum(),          # 33
      (x.f1 > b.f1.reindex(x.index)).sum())                                             # 25
```

**Interpolation predictions identical in LOSO and GroupKFold5, and fold composition** (reads the predictions files and the prepared dataset, ignored by git):

```python
import pandas as pd, numpy as np
o = pd.read_parquet("outputs/bench/oof/ca706e27654e0cf5/baseline_interpolacion_factor.parquet")
a = o[o.protocol == "LOSO"].sort_values("sample_idx"); b = o[o.protocol == "GroupKFold5"].sort_values("sample_idx")
print(np.abs(a.y_pred_reg.to_numpy() - b.y_pred_reg.to_numpy()).max())                # 0.0
P = "outputs/bench/prepared_v4_21/ca706e27654e0cf5/"
f = pd.read_parquet(P + "folds.parquet"); k = pd.read_parquet(P + "keys.parquet")
t = f[(f.protocol == "GroupKFold5") & (f.split == "test")].merge(
    k.reset_index(drop=True).rename_axis("sample_idx").reset_index(), on="sample_idx")
print(t.groupby("fold_id").factor_mult.unique().apply(lambda v: sorted(np.round(v, 1))))
```

**Perfect predictor on the real data** (`tabular_v4_21`):

```python
import pandas as pd
from swmm_resilience.ml.bench.evaluate import score_predictions
from swmm_resilience.ml.bench.schemas import OOF_COLUMNS, PreparedDataset
prepared = PreparedDataset.load("outputs/bench/prepared_v4_21/ca706e27654e0cf5")
t = prepared.folds[prepared.folds["split"] == "test"]; idx = t["sample_idx"].to_numpy()
y = prepared.y_clf.to_numpy()[idx]
oof = pd.DataFrame({"sample_idx": idx, "protocol": t["protocol"].to_numpy(), "fold_id": t["fold_id"].to_numpy(),
                    "y_pred_clf": y, "y_prob_clf": y.astype(float), "y_pred_reg": prepared.y_reg.to_numpy()[idx]},
                   columns=list(OOF_COLUMNS))
for p, m in score_predictions(prepared, oof, {"prep_id": prepared.prep_id}).items():
    print(p, m["classifier"]["f1"], m["classifier_legacy_mean_zero_fill"]["f1"], m["classifier_per_fold"]["f1"]["n_undefined"])
# GroupKFold5 1.0 1.0 0 · LOSO 1.0 0.8 5 · LeaveOneShapeOut 1.0 1.0 0
```

---

## 6. Full per-shape table — XGBoost and nearest shape, LeaveOneShapeOut

Source: `datos/por_forma.csv` (columns `n_rows`, `n_positives`, `f1`, `rmse_vol`, `error_vol_pct`). The column "% of squared error" is derived: `rmse_vol² × n_rows` of the shape divided by the sum of the 40 shapes (command of §5). Sorted by that column. Shapes with fewer than 4,000 rows lost cells to the continuity filter.

| # | Shape | Rows | Positives | XGBoost F1 | XGBoost RMSE (m³) | XGBoost total vol. error (%) | % of squared error (XGBoost) | Nearest-shape RMSE (m³) | Nearest-shape total vol. error (%) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `shape_quasi_uniform_block` | 4,000 | 1,206 | 0.9925 | 557.92 | −52.1 | 71.63 | 647.68 | −61.3 |
| 2 | `shape_sustained_plateau_14h` | 4,000 | 1,201 | 0.9950 | 160.34 | −18.5 | 5.92 | 211.84 | −28.0 |
| 3 | `shape_twin_plateau_15h` | 4,000 | 1,186 | 0.9945 | 130.26 | 11.8 | 3.90 | 148.29 | 19.3 |
| 4 | `long_moderate` | 4,000 | 1,184 | 0.9950 | 95.39 | 7.8 | 2.09 | 141.93 | −15.4 |
| 5 | `shape_triple_peak_long_decreasing` | 4,000 | 1,163 | 0.9881 | 88.85 | 14.6 | 1.82 | 221.31 | −48.8 |
| 6 | `shape_slow_broad_rise` | 4,000 | 1,186 | 0.9962 | 85.35 | −15.7 | 1.68 | 647.68 | 158.2 |
| 7 | `shape_gentle_broad_long` | 4,000 | 1,192 | 0.9971 | 83.67 | −11.2 | 1.61 | 141.93 | 18.2 |
| 8 | `shape_double_peak_equal_separated` | 4,000 | 1,183 | 0.9911 | 82.94 | −25.0 | 1.58 | 181.06 | −53.4 |
| 9 | `shape_central_flat_plateau` | 4,000 | 1,209 | 0.9883 | 71.01 | −24.3 | 1.16 | 78.33 | −29.0 |
| 10 | `shape_double_peak_intensifying` | 4,000 | 1,174 | 0.9949 | 68.76 | −22.0 | 1.09 | 50.49 | −10.3 |
| 11 | `shape_five_pulse_8h5` | 4,000 | 1,140 | 0.9828 | 60.98 | 20.5 | 0.86 | 103.75 | 48.2 |
| 12 | `common_storm` | 4,000 | 1,172 | 0.9932 | 55.45 | 19.4 | 0.71 | 78.33 | 40.8 |
| 13 | `shape_5h` | 4,000 | 1,190 | 0.9945 | 53.77 | 18.3 | 0.67 | 18.84 | −10.1 |
| 14 | `shape_recession_rebound` | 4,000 | 1,150 | 0.9820 | 48.82 | 15.5 | 0.55 | 160.98 | −57.5 |
| 15 | `new_hydrograph_2` | 4,000 | 1,195 | 0.9933 | 47.93 | −23.1 | 0.53 | 63.59 | −31.2 |
| 16 | `shape_back_loaded_late_peak` | 4,000 | 1,172 | 0.9974 | 47.05 | 29.9 | 0.51 | 78.55 | 62.2 |
| 17 | `shape_early_peak_secondary_shoulder` | 4,000 | 1,148 | 0.9828 | 44.58 | 18.4 | 0.46 | 99.27 | 61.0 |
| 18 | `shape_early_broad_6h45` | 4,000 | 1,177 | 0.9962 | 42.52 | −12.6 | 0.42 | 99.27 | −37.9 |
| 19 | `shape_four_pulse_cluster` | 4,000 | 1,126 | 0.9770 | 40.06 | 33.3 | 0.37 | 24.64 | 14.9 |
| 20 | `shape_triple_peak_short` | 4,000 | 1,154 | 0.9764 | 38.22 | 23.2 | 0.34 | 115.40 | 75.6 |
| 21 | `shape_terminal_peak_7h` | 4,000 | 1,172 | 0.9953 | 37.84 | −12.1 | 0.33 | 41.36 | −17.9 |
| 22 | `shape_delayed_abrupt_peak` | 4,000 | 1,159 | 0.9906 | 37.83 | 16.7 | 0.33 | 41.36 | 21.7 |
| 23 | `shape_double_peak_close` | 4,000 | 1,151 | 0.9807 | 34.76 | −15.5 | 0.28 | 15.12 | −2.4 |
| 24 | `shape_symmetric_moderate` | 4,000 | 1,160 | 0.9923 | 31.64 | −21.6 | 0.23 | 42.17 | −32.0 |
| 25 | `shape_late_peak_4h` | 4,000 | 1,170 | 0.9966 | 28.51 | −18.5 | 0.19 | 24.64 | −13.0 |
| 26 | `shape_long_recession_tail` | 4,000 | 1,158 | 0.9746 | 26.97 | 23.4 | 0.17 | 79.29 | 84.4 |
| 27 | `shape_front_loaded_broad` | 4,000 | 1,177 | 0.9970 | 21.94 | −6.7 | 0.11 | 15.12 | 2.5 |
| 28 | `shape_flashy_long_tail_9h5` | 4,000 | 1,162 | 0.9906 | 20.57 | −6.8 | 0.10 | 79.29 | −45.8 |
| 29 | `shape_short_flat_top` | 3,840 | 1,210 | 0.9838 | 18.76 | −19.8 | 0.08 | 7.15 | 2.4 |
| 30 | `base` | 3,840 | 1,187 | 0.9879 | 18.37 | −4.3 | 0.07 | 28.99 | −27.5 |
| 31 | `new_hydrograph` | 3,840 | 1,225 | 0.9484 | 18.00 | 19.6 | 0.07 | 28.99 | 37.8 |
| 32 | `shape_2h` | 3,840 | 1,196 | 0.9900 | 14.10 | 2.1 | 0.04 | 7.15 | −2.3 |
| 33 | `shape_offset_double_peak_2h5` | 3,840 | 1,129 | 0.9813 | 13.95 | 6.7 | 0.04 | 43.04 | 47.0 |
| 34 | `shape_sharp_peak_2h15` | 3,840 | 1,138 | 0.9844 | 12.15 | 2.6 | 0.03 | 13.15 | −16.3 |
| 35 | `shape_stepped_plateau_1h45` | 3,840 | 1,128 | 0.9809 | 11.33 | 10.5 | 0.03 | 13.15 | 19.5 |
| 36 | `shape_micro_burst_45min` | 3,360 | 1,139 | 0.9608 | 6.39 | −13.7 | 0.01 | 5.18 | −16.0 |
| 37 | `common_bogota` | 3,200 | 1,056 | 0.9188 | 5.32 | 31.6 | 0.01 | 70.55 | 436.9 |
| 38 | `shape_flash_flood_urban` | 3,360 | 1,137 | 0.9642 | 4.40 | −5.4 | 0.00 | 13.65 | −53.8 |
| 39 | `rare_extreme` | 2,880 | 1,005 | 0.9316 | 3.25 | −18.1 | 0.00 | 14.74 | 116.8 |
| 40 | `shape_ultra_short_impulse` | 2,400 | 858 | 0.8798 | 2.38 | 19.5 | 0.00 | 28.25 | 341.4 |
