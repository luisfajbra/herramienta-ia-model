# Annex 02 — ML and evaluation protocol

**Audit:** thesis project `herramienta-ia-model` (ML surrogate model of nodal flooding on SWMM)
**Branch / base commit:** `fix/cierre-auditoria-hidrogramas-20260927` @ `6e8579b`, with uncommitted changes of "Phase A" from another task (not touched)
**Date:** 2026-10-10
**Area:** data leakage, validation protocols (LOSO / GroupKFold5 / LeaveOneShapeOut), classifier→regressor cascade, metrics and their aggregation, decision threshold, uncertainty, reproducibility, model promotion, train/predict contracts and test coverage of `tests/ml`.

> Severity convention: 🔴 CRITICAL (compromises scientific validity) · 🟠 MAJOR (biases results) · 🟡 MODERATE (limitation or risk) · 🔵 MINOR (quality).
> Every number in this annex comes from a script executed in this audit (section 5) or from a file that was read. What was not verified is marked **not verified**.

---

## 1. Scope and method

### 1.1 Code reviewed

| Module | Review |
|---|---|
| `swmm_resilience/ml/bench/` (schemas, folds, preprocess, quality, train, evaluate, metrics, ranking, reports, promote, registry, param_filter, `models/*`) | Line by line, in the current working tree. That is, **with** the uncommitted Phase A changes in `folds.py`, `preprocess.py`, `quality.py` and `schemas.py`. |
| `swmm_resilience/ml/` (trainer, evaluator, preprocessing, contracts, predict, scenario_predict, feature_analysis, feature_importance, predict_tabular) | Line by line. |
| `swmm_resilience/ml/train.py` (*legacy* desktop stack, 1,116 lines) | Only the protocol parts: features, split, PCA, CV. |
| `swmm_resilience/ml/temporal/` (dataset, train_surrogate, train_cnn, predict, compare_surrogate) | Protocol, features, scalers and model selection. There are no temporal data or artifacts in the tree (it was verified that `data/training/swmm_resilience.db` does not exist), so no result could be contrasted. |
| `main.py` (`_run_bench`, `--only-ml`, `--evaluate-*`, `--analyze-features`), `swmm_resilience/config.py`, `bench:` block of `config.yaml` | CLI wiring, promotion and writing to `outputs/models`. |
| `tests/ml/**` and `tests/conftest.py` (`csv_shaped_dataset*` fixtures) | It was judged whether each test would really fail if the behavior were broken. |
| `swmm_resilience/database/csv_backfill.py::persist_training_run` | Only as a source of the OOF that already exist in SQL (section 4). |

### 1.2 Empirical diagnostics (all read-only)

- Data: `outputs/bench/prepared/af3751551ca5df98` is the current PreparedDataset: 160,000 rows, 1,000 runs, 160 nodes, 40 shapes, 25 factors and the 3 protocols. `4a48359e7e5b53c6` is the previous one (76,000 rows, 19 shapes) and is the one that produced the reports in `outputs/bench/reports/`. Both were read directly from the parquet files.
- SQLite `outputs/training_v17.sqlite3`: opened only with `file:…?mode=ro` (table `oof_predictions`).
- The **XGBoost OOF predictions were regenerated** with the project function `evaluate_candidate()` and the hyperparameters of `config.yaml`, on `af3751551ca5df98` and the 3 protocols (70 folds, 181.7 s). It is a single family, not the full benchmark. The OOF were left in the scratchpad and not in `outputs/`.
- Small controls: two no-learning baselines (lookup table), a perfect predictor scored with `score_predictions()`, `predict` versus `predict_proba` coherence per family (subsamples of 4,000/3,000 rows) and a regressor capacity control (flooded rows only, 5 folds).
- Tests: `pytest tests/ml/bench/{test_metrics,test_score_predictions,test_evaluate_candidate,test_parity_with_legacy_evaluator,test_folds}.py --basetemp=<scratchpad>` gave **51 passed**. The full suite was not run.
- No code, `config.yaml`, `outputs/` or git was modified. The scripts are in the scratchpad with prefix `m_` and are reproduced in section 5.

### 1.3 How to read the protocols (verified)

| Protocol | Group column (`schemas.py:56-60`) | Actual folds | What it actually measures |
|---|---|---|---|
| `LOSO` | `factor_mult` | 25 × 6,400 rows. 5 folds (factors 0.2–1.0) **with no flooded row at all** | Interpolation in the factor: the same node and the same shape are in train at ±0.2 |
| `GroupKFold5` | `factor_mult` | 5 × 32,000, interleaved factors (e.g. {1.0, 2.0, 3.0, 4.0, 5.0}) | The same as LOSO, with 5 held-out factors per fold |
| `LeaveOneShapeOut` (Phase A only, uncommitted) | `shape_id` | 40 × 4,000 | Generalization to a new shape, with all factors and nodes seen |

In no protocol are nodes held out or the new shape × new factor combination. Every generalization claim is bounded to **this network and these 160 nodes**.

---

## 2. Findings

### 2.1 Summary table

| ID | Sev. | File:line | Finding | Quantified impact |
|---|---|---|---|---|
| M-01 | 🔴 | `ml/bench/schemas.py:56-58`, `ml/evaluator.py:58,147`, `config.yaml:48` | LOSO/GroupKFold5 by factor only measure interpolation on a dense grid, and the bench has no baselines. A no-learning lookup table beats XGBoost | LOSO: the baseline obtains pooled e2e RMSE **9.0 m³** versus **68.4 m³** for XGBoost, and F1 0.972 versus 0.966 |
| M-02 | 🔴 | `ml/contracts.py:46-49` | The shape features (only `duracion_horas`, `tiempo_al_pico_h`) do not represent the hydrograph volume. With unseen shapes, the volume fails systematically | LeaveOneShapeOut: oracle NSE 0.814 (versus 0.953 in LOSO); total-volume error per shape from −63% to +216%; a single shape concentrates 61% of the SSE |
| M-03 | 🟠 | `ml/bench/metrics.py:41-43`, `evaluate.py:94,119`, `evaluator.py:88,127` | F1/precision/recall are averaged per fold with `zero_division=0`, so folds without positives count as 0 | In LOSO an F1 of 0.7408 is reported when the pooled value is 0.9656. A **perfect** predictor obtains F1 = 0.80 |
| M-04 | 🟠 | `evaluate.py:121`, `ranking.py:97`, `main.py:207` | The ranking metric is the unweighted mean of the per-fold RMSE, with no CI or paired comparison | LeaveOneShapeOut: mean 79.7 versus pooled 136.1 m³; group 95% CI of the pooled RMSE [66.7, 212.2] |
| M-05 | 🟠 | `ml/bench/train.py:85`, `evaluate.py:170`, `config.yaml:62-63` | The regressor is underfit and also has the `expm1(log1p)` retransformation bias | In-sample RMSE 118 m³ ≈ OOF 127 m³; total volume systematically −6%; small volumes +35%/+49% |
| M-06 | 🟠 | `config.yaml:56-90`, `models/random_forest_family.py:15` | The families are compared with fixed hyperparameters inherited from the previous winner, without tuning | RF `max_depth=6`: oracle NSE 0.45 (19 shapes). The ranking compares configurations, not methods |
| M-07 | 🟠 | `main.py:1082`, `ml/trainer.py:78-80`, `bench/promote.py:39-41` | `outputs/models` is silently overwritten by the default pipeline, and `promotion.json` is left false | SHA of the actual `classifier.joblib` (`660be918…`) ≠ that of `promotion.json` (`e2e7925e…`); the model in use was trained with 40 shapes and the promoted one was from 19 |
| M-08 | 🟡 | `main.py:198`, `bench/reports.py:54-75` | The bench does not persist the OOF. The reports carry no provenance, and the ranking is partially overwritten | The current `ranking.csv` has only `svm` (rank 1 of 1); the `metrics_*.json` come from different runs and carry no `prep_id` |
| M-09 | 🟡 | `evaluate.py:163`, `models/svm_family.py:30-32`, `bench/train.py:92-110` | The 0.5 threshold is implicit, the probabilities are not calibrated because of the reweighting and in SVM `predict` ≠ `predict_proba` | SVM: 2.5% disagreement; in the 0.4–0.5 bin the actual rate is 0.18–0.27 |
| M-10 | 🟡 | `ml/bench/metrics.py:36-70` | Missing metrics: confusion matrix, specificity, MCC/BA, PR-AUC, e2e MAE, relative volume error and per-shape breakdown | The LOSO AUC averages 20 folds while F1 averages 25 |
| M-11 | 🟡 | `evaluate.py:82-123` | Few independent units (25 factors / 40 shapes) and correlated nodes, without uncertainty | Group 95% CI of the pooled F1 in LOSO: [0.951, 0.975] |
| M-12 | 🟡 | `ml/feature_analysis.py:109-111,227-228,270-271`, `feature_importance.py:13` | The shape ablation is done with LOSO by factor. SHAP and importances assume a tree without scaler | The ablation cannot measure the shape contribution; if SVM/MLP/linear is promoted, `--analyze-features` fails or reports incorrectly |
| M-13 | 🟡 | `main.py:480-555, 557-640, 664-726` | `--evaluate-shapes` evaluates on the training data itself without labeling it; `--evaluate-generalization` is interpolation | The printed F1/NSE are training metrics (25 factors × trained shapes) |
| M-14 | 🟡 | `models/*_family.py` (`SimpleImputer(strategy="median")`) | Median imputation of structural missing values assigns upstream capacity to headwater nodes | 57,000 rows (35.6%): `upstream_capacity_lps` goes from NaN (=0 physical) to 110.6 L/s |
| M-15 | 🟡 | `docs/superpowers/specs/2026-09-29-optuna-…-design.md` §3-4 | The planned tuning is not nested: GroupKFold by shape over the whole dataset and the report is afterwards made on the same data | Risk of overfitting to the CV (not implemented in this branch) |
| M-16 | 🟡 | `ml/temporal/dataset.py:34`, `temporal/predict.py:493,546`, `train_cnn.py:215`, `train_surrogate.py:165,291` | Temporal surrogate: uses a SWMM output as input, has a train/inference mismatch, selects the model with the validation fold and the final loss ≠ the CV one | Not quantifiable (there are no data or artifacts in the tree) |
| M-17 | 🔵 | `tests/ml/bench/test_evaluate_candidate.py:57,79`, `test_score_predictions.py:87`, `test_parity_with_legacy_evaluator.py` | Tests that would not fail if the critical behavior were broken | See detail |
| M-18 | 🔵 | `XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md:569`, `config.yaml:61`, `evaluate.py:106` | The name "LOSO" is ambiguous ("Scenario" ≠ factor); there is an obsolete comment and a key `f"{factor:.2f}"` | Risk of misreading in the thesis |
| M-19 | 🔵 | `extraction/labels.py:85`, `quality.py` (commit) | The volume is quantized to 1 m³ and the 1.0 m³ threshold coincides with the resolution | 1,902 rows are exactly 1 m³; the median FN volume is 1 m³ |
| M-20 | 🔵 | `ml/predict.py:72`, `ml/scenario_predict.py:136` | At inference the feature contract is not validated, and the extrapolation flag ignores the shape | A NaN in a required feature would be silently imputed |
| M-21 | 🔵 | `database/csv_backfill.py:248,295`; `ml/train.py` | The SQL evidence stores `hyperparameters_json='{}'`; the legacy desktop stack uses another protocol (PCA, `run_id`) | Incomplete provenance |
| M-22 | 🔵 | `models/mlp_family.py:68` | Global `np.random.seed` as a side effect; torch determinism is not enforced | Not verified empirically |

**Count:** 🔴 2 · 🟠 5 · 🟡 9 · 🔵 6 (22 findings).

---

### 2.2 Finding detail

#### M-01 🔴 — The per-factor protocols measure trivial interpolation and there are no baselines

- **Where:** `PROTOCOL_GROUP_COLUMNS` (`ml/bench/schemas.py:56-58`) groups `LOSO` and `GroupKFold5` by `factor_mult`, like the legacy code (`ml/evaluator.py:58,147`). The dynamic features are exactly `base × factor`: `max|q_pico_nodo − base_inflow_lps·factor| = 1.4e-14` (m_01). When one factor is held out, the same node with the same shape is in train at ±0.2.
- **Evidence (m_06, on the current PreparedDataset):** for each (node, shape), the baseline takes the volume linearly interpolated between the neighboring factors. It learns nothing.

  | LOSO (factor) | Pooled F1 | Recall | Pooled e2e RMSE (m³) | Total vol. error |
  |---|---|---|---|---|
  | Interpolation baseline | **0.9718** | 0.9909 | **9.00** | −0.39% |
  | XGBoost (`config.yaml`) | 0.9656 | 0.9808 | 68.43 | −6.75% |

  At interior factors, the RMSE is 4.88 (baseline) versus 64.99 (XGBoost). In LeaveOneShapeOut, the baseline "nearest shape in (dur, t_pico), same node and factor" obtains F1 0.9758 versus 0.9789 and pooled RMSE 170.2 versus 136.1. In the per-shape mean it is 98.2 versus 79.7, and the baseline wins in **20 of 40 shapes**.
- **Impact:** LOSO and GroupKFold5 provide no evidence of generalization. A good number in LOSO is compatible with a model worse than a table. On unseen shapes, the ML gives a modest improvement (≈20% of mean RMSE) over the nearest neighbor. The thesis cannot assert the value of the surrogate without these baselines.
- **Recommendation:** (1) add to the bench a no-learning baseline "family" (interpolation by factor, nearest-shape neighbor and "always 0") that goes through `score_predictions()`; (2) declare LOSO/GroupKFold5 as **interpolation** and use LeaveOneShapeOut, together with a more demanding per-shape GroupKFold, as the main protocol; (3) add a combined protocol (held-out shape × factor, or blocks of extreme factors) to really measure extrapolation.

#### M-02 🔴 — Generalization to new shapes: the shape representation is insufficient

- **Where:** `ml/contracts.py:46-49`. The shape enters only through `duracion_horas` and `tiempo_al_pico_h`. The peak is already normalized, so `q_pico_*` does not change between shapes. No feature carries the hydrograph **volume or area**.
- **Evidence (regenerated OOF, m_02/m_03/m_04/m_05/m_07):**
  - Pooled oracle regressor: NSE **0.814** in LeaveOneShapeOut versus 0.953 in LOSO; RMSE 253 versus 127 m³; MAE 100.8 versus 49.6 m³.
  - Total-volume error per shape: from **−63.3%** (`shape_quasi_uniform_block`) to **+216.3%** (`common_bogota`). `shape_quasi_uniform_block` alone contributes **61.35%** of the end-to-end SSE.
  - The actual total volume per shape correlates with the area of the normalized hydrograph (Spearman **0.976**). The error per shape correlates with the fill factor `area/duration` (Spearman **−0.577**): the "block" shapes are underestimated and the "peaked" ones overestimated.
  - Capacity control (flooded rows only, GroupKFold5): a higher-capacity regressor lowers the RMSE from 127.5 to 57.5 m³ when the groups are factors, but from 327.7 to 317.0 m³ when they are shapes (NSE 0.69 → 0.71). **More capacity does not solve the problem with unseen shapes**: the bottleneck is the representation.
  - With 8 held-out shapes per fold (per-shape GroupKFold), the oracle NSE falls to 0.69. LeaveOneShapeOut, with 39 shapes in train, is the optimistic extreme.
- **Impact:** any claim about new hydrographs (the objective of the hydrograph validation) is unsupported in volume. The classification does generalize well (pooled F1 0.979).
- **Recommendation:** implement and evaluate with an ablation **under LeaveOneShapeOut** the tributary volume and integral variables already proposed in `docs/justificacion-features-tributarias-tabular-v4-21.md` (not implemented). Report the errors per shape and treat shapes outside the range (dur, t_pico, area) as extrapolation.

#### M-03 🟠 — Aggregation of F1/precision/recall with `zero_division=0` per fold

- **Where:** `ml/bench/metrics.py:41-43` (`zero_division=0`), `evaluate.py:94,119` (per-fold mean) and the same logic in `ml/evaluator.py:88,127`. It also affects `by_factor` (`evaluate.py:104-116`, `evaluator.py:122`).
- **Evidence:**
  - m_08: a **perfect predictor** scored with `score_predictions()` on the real PreparedDataset obtains, in LOSO, `precision = recall = f1 = 0.80` (AUC 1.0, RMSE 0). Moreover, `by_factor["0.20"].f1 = 0.0` in **all three protocols**.
  - m_03 (XGBoost): in LOSO, reported F1 0.7408, mean F1 over the 20 folds with positives 0.9260 and **pooled** F1 **0.9656**. The reported precision is 0.7226 and the pooled one 0.9508. The 5 folds without positives have 0 positive predictions: there the classifier is right and still scores 0.
  - The LOSO versus GroupKFold5 "gap" in the reports (0.727 versus 0.963 in `outputs/bench/reports/metrics_xgboost.json`) is an **artifact**: pooled gives 0.9656 versus 0.9659. The same value 0.7408 appears in `outputs/metrics/metrics_classifier.json` (legacy, 2026-10-03).
- **Impact:** it underestimates the LOSO F1, precision and recall by ≈0.22 and suggests a difference between protocols that does not exist. It affects all families equally (the ranking order changes little), but the absolute values reported are wrong.
- **Recommendation:** compute the pooled confusion matrix (sum of TN/FP/FN/TP per protocol) as the main metric. Per fold, use `zero_division=np.nan` and report how many folds have the metric defined. In `by_factor`, return NaN where there are no positives.

#### M-04 🟠 — Ranking metric: unweighted per-fold mean, with no uncertainty or paired test

- **Where:** `evaluate.py:121` (mean of `rmse_vol_todos_nodos` per fold), `ranking.py:97` (sorts by point value), `main.py:207` (protocol = `protocols[0]`, today `LeaveOneShapeOut` in Phase A).
- **Evidence (m_03):** in LeaveOneShapeOut, the per-fold mean is 79.7 m³ and the pooled 136.1 m³ (−41%). In LOSO, the mean is 55.3 (with 5 folds of RMSE 0) and the pooled 68.4. The e2e SSE comes in ~100% from TP rows (the classifier errors carry no weight in volume) and one shape dominates it (61%). The group-bootstrap 95% CI of the pooled RMSE in LeaveOneShapeOut is **[66.7, 212.2]**. In the old reports, `svm` beats `xgboost` by 37.70 versus 39.97 (≈6%), a difference well within that dispersion.
- **Impact:** the winner depends on how it is aggregated and on one or two shapes, and the ranking cannot tell a tie from a real difference.
- **Recommendation:** declare the aggregation (per-group mean for "typical new shape" or pooled for "total volume"), report both with group-bootstrap CI and decide between families with **per-fold paired** differences (bootstrap of the difference or Wilcoxon). Add MAE and relative volume error.

#### M-05 🟠 — Underfit regressor and log retransformation bias

- **Where:** `ml/bench/train.py:85` and `evaluate.py:169-170` (MSE in `log1p` and back with `expm1` without correction). Hyperparameters in `config.yaml:62-63`.
- **Evidence:** the **in-sample** (training) RMSE of the `config.yaml` regressor is 118.0 m³ with NSE 0.960, practically equal to the LOSO OOF (127.2 m³): the error is dominated by bias and not by the generalization gap. A higher-capacity regressor drops to 16.7 m³ in-sample and to 57.5 m³ out-of-sample by factor (m_07). `Σpred/Σreal` = 0.956 in-sample and 0.93–0.94 in the three protocols, so the **total volume is underestimated by ≈6%** systematically. By quantiles: volumes ≤24 m³ are overestimated by +35% (LOSO) and +49% (LeaveOneShapeOut), and those >822 m³ are underestimated between −8% and −16%.
- **Impact:** the volume metrics and the total-volume curve inherit a known and correctable bias.
- **Recommendation:** apply a retransformation correction (Duan smearing, estimated **within the fold**) or an objective on the original scale (Tweedie/Poisson). Tune the capacity with nested CV (see M-15) and report Σpred/Σreal.

#### M-06 🟠 — Comparison of families with fixed inherited hyperparameters

- **Where:** `config.yaml:56-90`. Those of XGBoost are "heredados del pipeline validado" (inherited from the validated pipeline); RF uses `max_depth=6` "para aislar el método" (to isolate the method) (`random_forest_family.py:15`). The MLP comments say "Pendiente de afinar" (Pending tuning).
- **Evidence:** with the 19-shape reports, RF obtains oracle NSE 0.446/0.449 and e2e RMSE 155–201, versus 0.965–0.977 and 37–47 for XGB/SVM. An RF with depth 6 is strongly constrained, unlike a boosting of depth 6, so equalizing the depth does **not** isolate the method.
- **Impact:** the ranking compares arbitrary configurations. Choosing the family before tuning it (the order set by the Optuna spec) favors the family whose configuration was inherited.
- **Recommendation:** give each family an equivalent tuning budget inside a nested CV before comparing, or declare the fixed benchmark as a "comparison of default configurations".

#### M-07 🟠 — `outputs/models` is silently overwritten and the promotion provenance is left false

- **Where:** the default pipeline (`python main.py`, `--only-ml`, `--skip-extraction`) calls `train_models(df, config, MODELS_DIR)` (`main.py:1082`), which writes `classifier.joblib`, `regressor.joblib` and `training_inp_hash.txt` (`ml/trainer.py:78-80`) without touching `promotion.json`. For its part, `promote_candidate` writes `training_inp_hash.txt` with the MD5 of the **current** `.inp` of `config.yaml`, not of the training one (`bench/promote.py:41`). The PreparedDataset manifest does not store the `.inp` hash either.
- **Evidence (read from disk):** `outputs/models/promotion.json` declares `family: xgboost`, `prep_id: 4a48359e7e5b53c6` (19 shapes) and `classifier_sha256: e2e7925e…` (2026-09-07). The present `classifier.joblib` has SHA-256 `660be918a9681677…` (2026-10-03) and `scale_pos_weight = 2.4600579559707625 = 113,758/46,242`, which is the balance of the 40-shape dataset. That is, it was retrained by the legacy pipeline.
- **Impact:** maps, `--evaluate-*`, SHAP and curves use a model that is not the bench candidate to which the metrics are attributed. `promotion.json` claims the opposite. The hash protection of `predict_network` certifies the `.inp` present at promotion time, not the training one.
- **Recommendation:** make `train_models` refuse to write to `outputs/models` when `promotion.json` exists, or invalidate it. Verify the SHAs of `promotion.json` against the binaries on load. Store in the candidate metadata the hash of the `.inp` with which the data were generated.

#### M-08 🟡 — The bench does not persist OOF and the reports carry no provenance

- **Where:** `main.py:198` discards the OOF frame (`_, metrics = evaluate_candidate(...)`). `reports.py:54-75` writes `metrics_<family>.json` without `prep_id`, hyperparameters, date or commit, and overwrites `ranking.*` only with the families of the current run.
- **Evidence:** the current `outputs/bench/reports/ranking.csv` contains **a single row** (`svm`, rank 1, 2026-09-07 02:16), while `metrics_xgboost/random_forest/mlp.json` are from other runs (20:28 and 23:03). All correspond to the old 19-shape PreparedDataset and there is no report for the current one (40 shapes, with LeaveOneShapeOut).
- **Recommendation:** persist `oof_<family>_<prep_id>.parquet` and include in each JSON the `prep_id`, hyperparameters, code version and date. Make the ranking read all the metrics of the same `prep_id` and never mix runs.

#### M-09 🟡 — Implicit decision threshold and uncalibrated probabilities

- **Where:** `evaluate.py:163` uses `classifier.predict` (fixed 0.5 threshold of each estimator) and inference does the same (`predict.py:73`, `scenario_predict.py:140`). The candidate metadata does not store the threshold (`bench/train.py:92-110`). SVC uses `probability=True` (`svm_family.py:32`).
- **Evidence:** in SVM, `predict` ≠ `predict_proba ≥ 0.5` in **75 of 3,000** rows (2.5%); in XGB/RF/linear/MLP there are 0 disagreements (control with 4,000 training rows). The reweighting (`scale_pos_weight` ≈ 2.46) miscalibrates: in XGBoost LOSO, the 0.4–0.5 probability bin has an actual rate of 0.27 and in LeaveOneShapeOut of 0.18. In the linear model, the mean probability is 0.356 versus an actual rate of 0.299. The threshold sweep in LeaveOneShapeOut gives a maximum pooled F1 ≈0.982 at 0.6 (0.979 at 0.5) and a recall of 0.996 at 0.3.
- **Impact:** the SVM F1 and its AUC correspond to different decision rules. The threshold is a design decision (FN cost versus FP) that is not explicitly made today.
- **Recommendation:** derive `y_pred = y_prob ≥ t` in all families, store `t` in the metadata and choose it in an inner CV (never on the report OOF). If the probabilities are going to be interpreted, calibrate them within the fold.

#### M-10 🟡 — Incomplete metrics for an imbalanced problem with a cascade

- **Where:** `ml/bench/metrics.py:36-70`.
- **Evidence:** only precision, recall, F1 and AUC of the positive class are reported. Missing are the confusion matrix, specificity, MCC or balanced accuracy, PR-AUC, the end-to-end MAE, the relative error of the total volume and the per-shape breakdown (only `by_factor` exists). `pct_nodos_correctos` (0.98) is dominated by the negatives (71%). In LOSO, the mean AUC excludes the 5 folds without positives (NaN), while F1 includes them as 0: the two metrics are averaged over different populations.
- **Recommendation:** see section 4. Add `by_shape` and relative volume MAE/bias.

#### M-11 🟡 — Few independent units and no uncertainty

- **Evidence:** the independent units are 25 factors and 40 shapes. The 160 rows of each run are nodes of the same network, hydraulically coupled. The group-bootstrap 95% CIs (m_03) are: pooled F1 in LOSO [0.951, 0.975], in GroupKFold5 [0.956, 0.974] and in LeaveOneShapeOut [0.970, 0.986]; pooled RMSE in LOSO [52.2, 82.0] and in LeaveOneShapeOut [66.7, 212.2]. The per-fold F1 ranges from 0.466 (factor 1.2) to 0.994.
- **Recommendation:** always report the CIs per group (not per row) and the number of groups. Do not interpret 160,000 rows as 160,000 independent observations.

#### M-12 🟡 — The shape ablation uses the wrong protocol; SHAP and importances assume a tree

- **Where:** `ml/feature_analysis.py:109-111` performs the ablation of `duracion_horas`/`tiempo_al_pico_h` with `LeaveOneGroupOut` by `factor_mult` and `np.nanmean` of F1 with `zero_division=0`, and uses the legacy `make_classifier` instead of the bench family. In `:227-228,270-271`, SHAP applies `named_steps["imputer"]` + `TreeExplainer(named_steps["model"])`. `feature_importance.py:13` uses `model.feature_importances_`.
- **Impact:** with the shape always seen in train, the ablation cannot measure what the shape features contribute to new shapes. If a non-tree family is promoted, `--analyze-features` fails (`TreeExplainer` on SVC) or computes SHAP on unscaled X. The latter is inference, **not verified** at runtime. Moreover, the report hides `n_tuberias_in` although the model uses it.
- **Recommendation:** perform the ablation with LeaveOneShapeOut and pooled metrics. Choose the explainer according to the family, or use permutation importance on the OOF.

#### M-13 🟡 — "Evaluation" modes that are not out-of-sample

- **Where:** `main.py:480-555` (`--evaluate-shapes`) evaluates the final model on **the training shapes × the 25 training factors** (`factors_to_eval = config.factors()`, `:514`) and prints F1/NSE without saying they are training metrics. `main.py:557-640` (`--evaluate-generalization`) uses shapes trained at midpoints (±0.1 of seen factors), i.e. interpolation (see M-01). `main.py:664-726` (`--evaluate-validation-shapes`) protects the out-of-sample condition only **by shape name** and labels "INTERPOLACIÓN/EXTRAPOLACIÓN" (INTERPOLATION/EXTRAPOLATION) only by duration.
- **Recommendation:** label the output (in-sample, interpolation or out-of-sample). In validation, compare the shape content (not the name) and compute the regime with (dur, t_pico, area).

#### M-14 🟡 — Median imputation of structural missing values

- **Evidence:** `diam_max_in`, `pendiente_max_in` and `upstream_capacity_lps` are NaN **if and only if** `n_tuberias_in == 0` (57,000 rows, 35.6%; verified). `SimpleImputer(median)` assigns them 0.2 m of diameter and 110.6 L/s of upstream capacity at headwater nodes, whose physical value is 0. Trees can compensate through `n_tuberias_in`, but linear, SVM and MLP receive a physically false value. Moreover, the imputer disables XGBoost's native NaN handling.
- **Recommendation:** impute with constant 0 in capacity and diameter, with `add_indicator=True`, or document the decision. Verify the effect with an ablation.

#### M-15 🟡 — The planned tuning (Optuna) is not nested

- **Where:** the spec `2026-09-29-optuna-tuning-familia-ganadora-design.md` §3-4 proposes GroupKFold(5) by `shape_id` over **the whole** PreparedDataset, without holdout, after choosing the family with the fixed benchmark. It is not implemented in this branch (it only exists in a separate worktree). The spec itself labels the result "exploratorio" (exploratory).
- **Risk:** if the tuned hyperparameters are afterwards reported with LeaveOneShapeOut on the same data, the estimate is optimistic (overfitting to the CV). With 40 shapes and a CI as wide as that of M-04, the risk is real.
- **Recommendation:** nested CV, with an outer LeaveOneShapeOut or per-shape GroupKFold loop and an inner per-shape loop over the outer train. As an alternative, reserve holdout shapes that are not touched until the end.

#### M-16 🟡 — Temporal surrogate (secondary line, no data in the tree)

- `temporal/dataset.py:34-37` declares `total_inflow_lps` as "PRE_SWMM", but it comes from pyswmm's `node.total_inflow` (`simulation/runner.py:485`), which includes the flow routed from upstream: **it is a SWMM output**. At inference, the series of the run with the lowest multiplier is scaled linearly (`temporal/predict.py:493,546`), something that does not reproduce the nonlinear routing with surcharge and flooding. This is a mismatch between train and inference.
- The CV is GroupKFold by `run_id` (`dataset.py:199,326`), with the shape and factor neighbors in train.
- `train_cnn.py:215` persists the model of the **best fold** according to the validation loss: there is selection bias and the model is trained with less data.
- `train_surrogate.py` evaluates in CV with `MSELoss` (`:291`) and trains the final model with `HuberLoss` and without scheduler (`:165`): the CV does not evaluate the deployed model.
- The docstring of `compare_surrogate.py:149` ("no SWMM outputs") is false because of what the first point says.
- **Impact:** not quantifiable (`data/training/swmm_resilience.db` and artifacts do not exist). **Not verified** with data.

#### M-17 🔵 — Test coverage: the critical behavior would not fail

- `test_evaluate_candidate.py:79` (`…a_fold_without_flooded_training_rows_still_produces_predictions`) **does not exercise** the `else` branch of `evaluate.py:171-180`. With the fixture (`tests/conftest.py:57-90`: factors 1, 2 and 3, with flooding from factor ≥2), every LOSO fold has positives in train. Verified by reading.
- `test_score_predictions.py:87` ("perfect predictions score perfectly") uses a fixture in which all folds have positives. With the real data, the same perfect predictor obtains 0.80 (m_08), and no test detects it.
- `test_parity_with_legacy_evaluator.py` **enshrines** the M-03 bias: both sides compute the zero-filled mean in the same way. Its LOSO fixture has a fold without positives (factor 1.0).
- `test_evaluate_candidate.py:57` only checks that the scaler is fit with fewer rows than the total. It does not verify that it uses exactly the fold's train rows, so a fit on the test would pass.
- There are no tests of threshold or calibration, of OOF persistence, of `predict`/`predict_proba` coherence, nor that `train_models` does not overwrite a promotion.
- **Recommendation:** add tests with a fold without positives that require NaN (not 0) and pooled F1 = 1 for the perfect predictor, plus a spy that compares the exact set of fitted indices.

#### M-18 🔵 — Naming and leftovers

- "LOSO" groups by factor, but the documentation says "Leave One Scenario Out" (`XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md:569`), whereas in the DB a `scenario` is (shape, factor). It should be renamed to "LeaveOneFactorOut" in the thesis.
- `config.yaml:61` mentions "175 escenarios" (175 scenarios), when today there are 1,000 runs.
- `by_factor` uses `f"{factor:.2f}"` (`evaluate.py:106`), which would collide with a grid finer than 0.01.

#### M-19 🔵 — Volume resolution and flooding

- The `.rpt` reports the volume in 10⁶ L with 3 decimals (×1000 in `labels.py:85`), so all volumes are **integers in m³** (verified: 0 non-integer values). 1,902 rows are exactly 1.0 m³, which is the threshold, so `inunda` is equivalent to "SWMM reported some flooding". The median FN volume is 1 m³ (LOSO): part of the classification error lies at the resolution limit.
- The committed `quality.py` used `>` and reported 1,072 (19 shapes) or 1,902 (40 shapes) false "disagreements". Phase A (uncommitted) fixes it with `flood_label` (`>=`). In the data, `inunda == (vol ≥ 1)` has 0 discrepancies.
- **Recommendation:** document it in the thesis and analyze separately the FN with volume ≤2 m³.

#### M-20 🔵 — Feature contract at inference

- `predict.py:72` and `scenario_predict.py:136` build `X = merged[FEATURE_COLS]` without going through `TABULAR_V3_17.validate_frame` (training does: `preprocess.py:185`, `trainer.py:64`). A NaN in a required feature would be silently imputed with the median. The columns and their order do match (same tuple `FEATURE_COLUMNS_V17`).
- `scenario_predict._extrapolation_flags` flags extrapolation only by the `q_pico/base` ratio, not by (dur, t_pico) outside the trained range.

#### M-21 🔵 — Provenance in SQL and legacy stack

- `persist_training_run` inserts `hyperparameters_json='{}'` into `training_runs` and `model_evaluations` (`csv_backfill.py:248,295`): the SQL evidence does not say with which hyperparameters it was trained.
- `ml/train.py` (desktop) uses another protocol: `GroupShuffleSplit` by `run_id`, PCA(5) and regression over all rows, with the features left after `ML_DROP_COLUMNS` (blacklist, not a contract). It is inactive because there are no data (`dataset_ml.csv` does not exist), but it should be isolated or retired so that it is not confused with the bench.

#### M-22 🔵 — Determinism of the MLP family

- `mlp_family.py:68` calls `np.random.seed(...)`, which alters the global RNG as a side effect. `torch.use_deterministic_algorithms` and the number of threads are not enforced. **Not verified** whether it gives bit-for-bit identical results across machines or threads.

---

## 3. What is fine (verified)

1. **There is no target leakage in `tabular_v3_17`.** `q_pico_nodo` and `q_pico_acum_escalado` are exactly `base × factor` (maximum error 1.4e-14 and 4.5e-13). `duracion_horas` and `tiempo_al_pico_h` come from the shape, and no feature uses SWMM outputs (m_01, `extraction/dynamic_features.py:20-25`).
2. **The preprocessing is fit within the fold.** Imputer and scaler live in each family's `Pipeline` and are fit with `X_train` (`evaluate.py:154-169`). `scale_pos_weight` is computed per fold (`evaluate.py:158-159`). Stage 1 does not import transformers (guarded by `test_stage1_imports.py`).
3. **Integrity of groups in the folds.** In the actual folds: LOSO 25×6,400, GroupKFold5 5×32,000 and LeaveOneShapeOut 40×4,000. Each sample is in test only once per protocol, and no group appears in train and test at the same time. The Phase A validation requires it on load (`schemas.py:170-362`). `build_folds` rejects groups from the wrong column (`folds.py:62-67`).
4. **Data without duplicates or conflicts.** There are 160,000 distinct X vectors out of 160,000, 0 vectors with conflicting labels, 0 (run, node) duplicates and 0 `inunda` versus `vol ≥ 1` discrepancies. There are no 2 shapes with the same (dur, t_pico) pair.
5. **Parity and determinism.** The LOSO metrics regenerated with the bench match **digit for digit** `outputs/metrics/*.json` of the legacy pipeline of 2026-10-03 (F1 0.74076924794143; e2e RMSE 55.304099293627814; oracle NSE 0.9530489319999711). This verifies the bench ≡ legacy equivalence on real data and the reproducibility of XGBoost with seed 42.
6. **The cascade is well separated.** The regressor is trained only with `inunda==1` of the fold's train. The oracle level uses actual labels and the end-to-end uses the predicted ones (`evaluate.py:96-102`), and the documentation explicitly says that the oracle is an optimistic bound. Negative predictions are clipped to 0 and the case without positives in train predicts 0 instead of failing.
7. **Data provenance.** `prep_id` hashes the full descriptor, including the canonical SHA-256 of the frame. Capture is done by read-only SQLite `backup`, and `score_predictions` rejects OOF from another `prep_id`.
8. **Promotion (Phase A, uncommitted).** `--bench` no longer promotes and `--bench-promote` requires naming the family (`main.py:213-219`, `config.yaml:55`). This closes the self-promotion of the base commit, where `promote: "auto"` copied the ranking winner to `outputs/models`.
9. The JSON reports are sanitized (NaN→null) and the ranking keeps the invalid candidates with their reason.

---

## 4. Confusion matrix: current state and proposal

### 4.1 What exists today

| Source | Per-row prediction? | Content | Protocols/families | Useful for the matrix? |
|---|---|---|---|---|
| `evaluate_candidate()` → OOF frame (`schemas.py:44-51`: `sample_idx, protocol, fold_id, y_pred_clf, y_prob_clf, y_pred_reg`) | Yes, in memory | `y_true` is obtained with `prepared.y_clf[sample_idx]` | All families and protocols | **Yes, but it is discarded**: `main.py:198` (`_`). Nothing is left on disk |
| `outputs/bench/reports/metrics_*.json` | No | Per-fold means of P/R/F1/AUC and `by_factor` (F1, RMSE) | 4 families × {LOSO, GroupKFold5} on 19 shapes | **No**: there are no TN/FP/FN/TP counts nor a way to reconstruct them |
| `outputs/metrics/metrics_*.json` (legacy) | No | The same, only main LOSO | XGBoost | No |
| SQLite `oof_predictions` (`--persist-sql`, `csv_backfill.py:284-370`) | **Yes** | `run_id, node_pk, observed, predicted, probability, fold_id` (160,000 `inunda` rows; the volume ones without probability) | **Only** XGBoost × GroupKFold5 by factor (`training_runs.grouping_strategy='group_kfold'`, 2026-09-15) | Yes, only for that protocol |

With the SQL table (m_09), the pooled GroupKFold5 matrix is **TN=111,468 · FP=2,290 · FN=920 · TP=45,322** (recall 0.9801, specificity 0.9799, F1 0.9658). That of the regenerated OOF (m_03) is TN=111,431 · FP=2,327 · FN=877 · TP=45,365. The small difference comes, presumably, from the row order versus XGBoost's `subsample`; this was **not verified**.

Pooled matrices of the regenerated XGBoost OOF (positive class = flooding; m_03):

| Protocol | TN | FP | FN | TP | Recall | Specif. | MCC | Actual vol. in FN |
|---|---|---|---|---|---|---|---|---|
| GroupKFold5 | 111,431 | 2,327 | 877 | 45,365 | 0.9810 | 0.9795 | 0.9520 | 7,368 m³ (0.05%) |
| LOSO | 111,412 | 2,346 | 890 | 45,352 | 0.9808 | 0.9794 | 0.9515 | 6,211 m³ (0.04%) |
| LeaveOneShapeOut | 112,153 | 1,605 | 373 | 45,869 | 0.9919 | 0.9859 | 0.9703 | 728 m³ (0.00%) |

The FN are small floods (median of 1–2 m³; maximum 327 m³). The actual weight of the classifier errors on the volume is negligible compared with that of the regressor (M-04).

### 4.2 What is missing

1. Persist the OOF frame per family and `prep_id`, adding `y_true_clf`, `y_true_reg`, `factor_mult` and `shape_id` so as not to depend on a join.
2. Compute in `score_predictions` the TN/FP/FN/TP counts per fold and pooled per protocol, and the `by_factor` and `by_shape` breakdowns.
3. Set the threshold explicitly and record it: today it depends on each estimator's `predict` (M-09).
4. Export the matrix (JSON/CSV) and, optionally, a figure.

### 4.3 Where it plugs in

| Step | File | Proposed change |
|---|---|---|
| Counts | `ml/bench/metrics.py` | `confusion_counts(y_true, y_pred) -> {tn, fp, fn, tp}`, together with derived metrics (recall, specificity, precision, F1, MCC, BA) computed **from summed counts** |
| Aggregation | `ml/bench/evaluate.py::score_predictions` (loop `:87-116`) | Accumulate counts per fold, sum them per protocol (pooled) and per factor and shape. Keep the per-fold metrics with `NaN` where they are undefined |
| Threshold | `ml/bench/evaluate.py:163-164`, `bench/train.py` (metadata) | `y_pred = (y_prob >= t)` with `t` in the config or the metadata. For SVM it removes the `predict`/`predict_proba` inconsistency |
| Persistence | `main.py:198`, `ml/bench/reports.py` | Save `oof` to `reports/oof_<family>_<prep_id>.parquet` and `confusion_<family>.json/csv` (pooled, per fold, per factor and per shape) |
| Schema | `ml/bench/schemas.py:44-51` | Keep `OOF_COLUMNS` as the minimum contract and add optional traceability columns (`threshold`) without breaking `_validate_oof` |
| Tests | `tests/ml/bench/test_score_predictions.py` | Perfect predictor with a fold without positives → diagonal matrix, pooled F1 1 and per-fold recall NaN (not 0) |

### 4.4 Pending decisions (for the author)

1. **Positive class:** `inunda = 1` (vol ≥ 1 m³, that is, any reported flooding; see M-19). Flooding recall and specificity are the metrics to name in the thesis.
2. **Threshold:** keep 0.5 on reweighted probabilities or choose `t` in an **inner** CV according to an explicit cost (e.g. maximize F1 or fix recall ≥ 0.99). Never choose it on the report OOF. For the current OOF, the threshold changes F1 by <0.005 between 0.4 and 0.7.
3. **Aggregation:** pooled (sum of counts) as the main figure and per-fold dispersion as uncertainty (group bootstrap CI). Folds without positives contribute valid TN/FP, but their recall is undefined (NaN), not 0.
4. **Normalization:** by actual row (recall/specificity, the natural reading of "what fraction of floods do I detect"), in addition to the raw counts. Column normalization (precision) goes as a complement.
5. **Stratification:** one matrix per protocol and family at minimum, plus one per factor (factors ≤1.0 are all TN/FP) and one per shape in LeaveOneShapeOut (the worst are `shape_ultra_short_impulse`, with FP=288, and `common_bogota`, with FP=178).
6. **Comparability:** before comparing families, unify the `y_prob ≥ t` rule across all of them (SVM).

---

## 5. Reproducible scripts

All are run from the repo root with the project's Python. They only read `outputs/` and write to the scratchpad. `S` is the session scratchpad (`…\288ddaf2-…\scratchpad`). Those that import `swmm_resilience` need `PYTHONPATH=.`.

| Script | What it does | Findings | Time |
|---|---|---|---|
| `m_01_datos.py <prepared>` | Structure, `q = base×factor` consistency, duplicates, threshold, folds without positives | M-01, M-03, M-14, M-19, §3 | <1 min |
| `m_02_oof_xgboost.py <prepared> <S> xgboost` | Regenerates the OOF with `evaluate_candidate()` and `config.yaml` (3 protocols) | Basis of M-02…M-05, §4 | 182 s |
| `m_03_analisis_oof.py <prepared> <oof>` | Mean versus pooled, confusion matrix, threshold, calibration, oracle by quantile, SSE decomposition, group bootstrap | M-02…M-05, M-09…M-11 | ~1 min |
| `m_04_por_grupo.py <prepared> <oof>` | Matrix and error per shape and per factor | M-02, M-04 | <1 min |
| `m_05_area_forma.py <prepared> <oof>` | Correlation of the per-shape error with the hydrograph area | M-02 | <1 min |
| `m_06_baselines.py <prepared> <oof>` | No-learning baselines (factor interpolation; nearest shape) | M-01 | ~1 min |
| `m_07_capacidad.py <prepared>` | Regressor capacity control by factor and by shape | M-02, M-05 | ~2 min |
| `m_08_oraculo_perfecto.py <prepared>` | Scores a perfect predictor with `score_predictions()` | M-03, M-17 | ~1 min |
| `m_09_matriz_sql.py outputs/training_v17.sqlite3` | Confusion matrix from `oof_predictions` (`mode=ro`) | §4 | <10 s |

Commands used:

```bash
S="<scratchpad>"; P=outputs/bench/prepared/af3751551ca5df98
venv/Scripts/python.exe $S/m_01_datos.py $P
PYTHONPATH=. venv/Scripts/python.exe $S/m_02_oof_xgboost.py $P $S xgboost
venv/Scripts/python.exe $S/m_03_analisis_oof.py $P $S/m_oof_xgboost_af3751551ca5df98.parquet
venv/Scripts/python.exe $S/m_04_por_grupo.py   $P $S/m_oof_xgboost_af3751551ca5df98.parquet
PYTHONPATH=. venv/Scripts/python.exe $S/m_05_area_forma.py $P $S/m_oof_xgboost_af3751551ca5df98.parquet
venv/Scripts/python.exe $S/m_06_baselines.py   $P $S/m_oof_xgboost_af3751551ca5df98.parquet
PYTHONPATH=. venv/Scripts/python.exe $S/m_07_capacidad.py $P
PYTHONPATH=. venv/Scripts/python.exe $S/m_08_oraculo_perfecto.py $P
venv/Scripts/python.exe $S/m_09_matriz_sql.py outputs/training_v17.sqlite3
venv/Scripts/python.exe -m pytest tests/ml/bench/test_metrics.py tests/ml/bench/test_score_predictions.py \
  tests/ml/bench/test_evaluate_candidate.py tests/ml/bench/test_parity_with_legacy_evaluator.py \
  tests/ml/bench/test_folds.py -q -p no:cacheprovider --basetemp=$S/m_pytest   # 51 passed
```

The full outputs were left in `S/m_0*_out*.txt`. The two scripts that support the critical findings are reproduced below, so that they do not depend on the scratchpad.

<details>
<summary><code>m_06_baselines.py</code> (M-01)</summary>

```python
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

d, oof_path = Path(sys.argv[1]), Path(sys.argv[2])
keys = pd.read_parquet(d / "keys.parquet"); X = pd.read_parquet(d / "features.parquet")
T = pd.read_parquet(d / "targets.parquet"); oof = pd.read_parquet(oof_path)
df = keys[["node_id", "shape_id", "factor_mult"]].copy()
df["f"] = df.factor_mult.round(4); df["v"] = T.vol_inundacion_m3.to_numpy(); df["y"] = T.inunda.to_numpy()
df["dur"] = X.duracion_horas.to_numpy(); df["tp"] = X.tiempo_al_pico_h.to_numpy()
piv = df.pivot_table(index=["node_id", "shape_id"], columns="f", values="v"); fs = np.array(sorted(piv.columns))
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
pred = np.empty(len(df)); lookup = {f: piv[f] for f in fs}
for i, (n, s, f) in enumerate(zip(df.node_id, df.shape_id, df.f)):          # LOSO: interpolación entre vecinos
    k = np.searchsorted(fs, f)
    if k == 0: pred[i] = lookup[fs[1]][(n, s)]
    elif k == len(fs) - 1: pred[i] = lookup[fs[-2]][(n, s)]
    else:
        lo, hi = fs[k - 1], fs[k + 1]; a, b = lookup[lo][(n, s)], lookup[hi][(n, s)]
        pred[i] = a + (b - a) * (f - lo) / (hi - lo)
o = oof[oof.protocol == "LOSO"].sort_values("sample_idx")
routed = np.where(o.y_pred_clf.to_numpy() == 1, o.y_pred_reg.to_numpy(), 0.0)
print("LOSO baseline F1=%.4f RMSE=%.2f | XGB F1=%.4f RMSE=%.2f" % (
    f1_score(df.y, pred >= 1), rmse(df.v, pred), f1_score(df.y, o.y_pred_clf), rmse(df.v, routed)))
sh = df[["shape_id", "dur", "tp"]].drop_duplicates().set_index("shape_id"); z = (sh - sh.mean()) / sh.std()
piv2 = df.pivot_table(index=["node_id", "f"], columns="shape_id", values="v")
nearest = {s: ((z.drop(index=s) - z.loc[s]) ** 2).sum(axis=1).idxmin() for s in sh.index}
pred2 = np.array([piv2.at[(n, f), nearest[s]] for n, s, f in zip(df.node_id, df.shape_id, df.f)])
o2 = oof[oof.protocol == "LeaveOneShapeOut"].sort_values("sample_idx")
routed2 = np.where(o2.y_pred_clf.to_numpy() == 1, o2.y_pred_reg.to_numpy(), 0.0)
print("LOSHO baseline F1=%.4f RMSE=%.2f | XGB F1=%.4f RMSE=%.2f" % (
    f1_score(df.y, pred2 >= 1), rmse(df.v, pred2), f1_score(df.y, o2.y_pred_clf), rmse(df.v, routed2)))
```
</details>

<details>
<summary><code>m_08_oraculo_perfecto.py</code> (M-03)</summary>

```python
import sys
import pandas as pd
from swmm_resilience.ml.bench.evaluate import score_predictions
from swmm_resilience.ml.bench.schemas import OOF_COLUMNS, PreparedDataset

prepared = PreparedDataset.load(sys.argv[1])
tests = prepared.folds[prepared.folds["split"] == "test"]; idx = tests["sample_idx"].to_numpy()
oof = pd.DataFrame({"sample_idx": idx, "protocol": tests["protocol"].to_numpy(),
                    "fold_id": tests["fold_id"].to_numpy(),
                    "y_pred_clf": prepared.y_clf.to_numpy()[idx],
                    "y_prob_clf": prepared.y_clf.to_numpy()[idx].astype(float),
                    "y_pred_reg": prepared.y_reg.to_numpy()[idx]}, columns=list(OOF_COLUMNS))
for p, m in score_predictions(prepared, oof, {"prep_id": prepared.prep_id}).items():
    print(p, m["classifier"], m["by_factor"].get("0.20"))
# Salida: LOSO {'precision': 0.8, 'recall': 0.8, 'f1': 0.8, 'auc_roc': 1.0}; by_factor 0.20 f1 = 0.0 en los 3 protocolos
```
</details>

<details>
<summary>Core of <code>m_03_analisis_oof.py</code> (mean versus pooled, matrix, group bootstrap)</summary>

```python
# por protocolo: po = oof[oof.protocol == p]; idx = po.sample_idx
per = [{"f1": f1_score(yc[i], pc, zero_division=0), "rmse": rmse(yr[i], routed_i), "n_pos": yc[i].sum()}
       for each fold]                                   # métrica por fold como el banco
f1_banco  = mean(per.f1)                                # lo que reporta score_predictions
f1_pooled = f1_score(yc[idx], po.y_pred_clf)            # sobre la concatenación
tn, fp, fn, tp = confusion_matrix(yc[idx], po.y_pred_clf, labels=[0, 1]).ravel()
# bootstrap por grupo: remuestrear folds (grupos) con reemplazo 1000 veces y recalcular F1/RMSE pooled
```
Full script: `S/m_03_analisis_oof.py`.
</details>
