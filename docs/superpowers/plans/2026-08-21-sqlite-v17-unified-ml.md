# SQLite V17 Unified ML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, evaluate, select, store, and load every approved tabular model through one SQLite-backed, exact-v17 implementation.

**Architecture:** A model registry builds leakage-safe sklearn pipelines; a metric registry computes and ranks OOF results; `training.py` persists folds, OOF predictions, metrics, and final selected models; `artifacts.py` verifies model BLOBs before deserialization; prediction reads models and source data by database ID.

**Tech Stack:** Python 3.11, sqlite3, pandas, NumPy, scikit-learn, XGBoost, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md`

## Global Constraints

- Requires Plans A and B phase gates.
- Every candidate receives identical `FEATURE_COLUMNS_V17`, rows, groups, and fold IDs.
- Imputer/scaler/target transform fit only on training folds.
- OOF metrics never use final full-data fit predictions.
- Metric direction and tie-breaking are registry data, not conditional CLI code.
- Model BLOB is hashed before storage and before deserialization.
- No filesystem fallback, compatibility fill, or 15-feature load path.

---

### Task 1: Add Structured Model And Selection Configuration

**Files:**
- Modify: `config.yaml`
- Modify: `swmm_resilience/config.py:61-160,202-285`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `ModelCandidateConfig`, `SelectionConfig`, and validated `Config.ml`
- Consumes: YAML model definitions

- [ ] **Step 1: Replace the ML test fixture with explicit candidates**

Use this configuration shape:

```yaml
ml:
  random_seed: 42
  cv_folds: 5
  models:
    regression:
      ridge: {enabled: true, alpha: 1.0}
      lasso: {enabled: true, alpha: 0.001, max_iter: 20000}
      svr_rbf: {enabled: true, C: 10.0, epsilon: 0.1, gamma: "scale"}
      random_forest: {enabled: true, n_estimators: 300, max_depth: null}
      xgboost: {enabled: true, n_estimators: 300, max_depth: 6, learning_rate: 0.05, subsample: 0.9, colsample_bytree: 0.9}
    classification:
      logistic_regression: {enabled: true, C: 1.0, max_iter: 5000}
      svc_rbf: {enabled: true, C: 10.0, gamma: "scale", probability: true}
      random_forest: {enabled: true, n_estimators: 300, max_depth: null}
      xgboost: {enabled: true, n_estimators: 300, max_depth: 6, learning_rate: 0.05, subsample: 0.9, colsample_bytree: 0.9}
  selection:
    classification:
      primary: "pr_auc"
      tie_breakers: ["f1", "recall"]
    regression:
      primary: "log_nse"
      tie_breakers: ["rmse", "mae"]
    system:
      primary: "total_volume_error_pct"
      tie_breakers: ["f1", "csi"]
  prediction:
    classification_threshold: 0.5
    allow_network_extrapolation: false
```

Tests reject an unknown model, unknown metric, empty enabled model family,
`cv_folds < 2`, and `probability: false` for the configured SVC because PR-AUC
requires scores. Reject a classification threshold outside `[0, 1]` and a
non-boolean extrapolation setting.

- [ ] **Step 2: Run config tests and observe failure**

```powershell
python -m pytest tests/test_config.py -q
```

- [ ] **Step 3: Implement frozen config dataclasses and validation**

Use these immutable shapes:

```python
@dataclass(frozen=True)
class ModelCandidateConfig:
    algorithm: str
    enabled: bool
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class SelectionConfig:
    primary: str
    tie_breakers: tuple[str, ...]


@dataclass(frozen=True)
class PredictionConfig:
    classification_threshold: float
    allow_network_extrapolation: bool


@dataclass(frozen=True)
class MLConfig:
    random_seed: int
    cv_folds: int
    models: Mapping[str, tuple[ModelCandidateConfig, ...]]
    selection: Mapping[str, SelectionConfig]
    prediction: PredictionConfig
```

Copy parameter and outer mappings into `MappingProxyType` values on load so
callers cannot mutate raw YAML. Remove the parallel `ML_MODEL_CONFIGS`, PCA
settings, and legacy target constants only in Plan D after all legacy
consumers migrate.

- [ ] **Step 4: Update root config and commit**

```powershell
python -m pytest tests/test_config.py -q
git add config.yaml swmm_resilience/config.py tests/test_config.py
git commit -m "config: declare unified tabular model candidates"
```

### Task 2: Implement The Model Registry

**Files:**
- Create: `swmm_resilience/ml/model_registry.py`
- Create: `tests/ml/test_model_registry.py`

**Interfaces:**
- Produces: `build_candidates(task, model_config, random_seed) -> dict[str, BaseEstimator]`
- Consumes: structured model config and task (`classification` or `regression`)

- [ ] **Step 1: Write parametrized construction tests**

```python
@pytest.mark.parametrize("task,names", [
    ("regression", {"ridge", "lasso", "svr_rbf", "random_forest", "xgboost"}),
    ("classification", {"logistic_regression", "svc_rbf", "random_forest", "xgboost"}),
])
def test_registry_builds_all_approved_candidates(config, task, names):
    assert set(build_candidates(task, config.ml.models[task], 42)) == names
```

For each linear/SVM candidate assert pipeline steps are
`imputer -> scaler -> estimator`. For tree candidates assert
`imputer -> estimator`. For regressors assert the returned estimator is a
`TransformedTargetRegressor` with `func=np.log1p` and `inverse_func=np.expm1`.

Fit every candidate to one small 17-column frame containing allowed nulls and
assert it predicts finite outputs. Assert a 15-column frame fails through an
explicit contract-checking entry point before sklearn sees it.

Add a two-fold leakage test with deliberately different missing-value medians
and feature means. Inspect each fitted fold pipeline and assert its imputer
statistics and linear/SVM scaler means equal only that fold's training rows,
not the full dataset or validation rows. Add a target column to the input frame
and assert contract validation rejects it rather than treating it as a feature.

- [ ] **Step 2: Implement task/model definitions**

```python
def _base_pipeline(estimator, *, scale: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", estimator))
    return Pipeline(steps)
```

Build Ridge/Lasso/SVR/Logistic/SVC from sklearn, Random Forest from
`sklearn.ensemble`, and XGBoost lazily so importing non-XGBoost workflows does
not fail if its binary is unavailable. Set every supported random seed.

Before each fold fit and final fit, reject any contract-approved nullable
feature whose training slice is entirely null. This prevents sklearn from
dropping the column or manufacturing an untraceable zero when no fold-local
median exists. Add a test for that failure and for a valid fold-local median.

- [ ] **Step 3: Add a contract-checked fit helper**

```python
def validated_xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = frame.loc[:, FEATURE_COLUMNS_V17]
    TABULAR_V3_17.validate_frame(X)
    y = pd.to_numeric(frame[target], errors="raise")
    if y.isna().any() or not np.isfinite(y).all():
        raise ValueError(f"Invalid target values for {target}")
    return X, y
```

For regression, the training caller filters to the approved flooded/oracle
population before calling this helper; the registry does not silently filter.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/ml/test_model_registry.py tests/ml/test_feature_contract_v17.py -q
git add swmm_resilience/ml/model_registry.py tests/ml/test_model_registry.py
git commit -m "feat: unify tabular model registry"
```

### Task 3: Implement The Configurable Metric Registry

**Files:**
- Create: `swmm_resilience/ml/metric_registry.py`
- Create: `tests/ml/test_metric_registry.py`

**Interfaces:**
- Produces: `MetricDefinition`, `METRICS`, `compute_metric()`, `rank_candidates()`
- Consumes: observed/predicted/probability arrays and configured metric order

- [ ] **Step 1: Write metric definition and ranking tests**

Test that:

```python
assert METRICS["pr_auc"].direction == "maximize"
assert METRICS["log_nse"].direction == "maximize"
assert METRICS["rmse"].direction == "minimize"
assert METRICS["total_volume_error_pct"].direction == "minimize"
```

Add tests for PR-AUC, F1, precision, recall, ROC-AUC, CSI, MAE, RMSE, NSE,
log-NSE, total volume error, TN/FP/FN/TP counts, undefined single-class
ROC-AUC, zero-observed volume percentage, and deterministic tie-breaks.

- [ ] **Step 2: Implement typed definitions**

```python
@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | None
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    task: str
    direction: Literal["maximize", "minimize"]
    requires_probability: bool
    compute: Callable[..., MetricResult]
```

Undefined metrics return `MetricResult(valid=False, reason=...)`; never store
NaN as if it were comparable.

Compute and persist metric scopes for the global OOF population and for each
scenario key, factor, hydrograph shape identity, and extrapolation state when
that metadata is present. A stratum with undefined inputs records an invalid
metric and reason; it is not dropped.

- [ ] **Step 3: Implement stable ranking**

`rank_candidates(rows, primary, tie_breakers)` must reject invalid primary
metrics, normalize minimize metrics only for sorting, apply tie-breakers in
order, and use algorithm name as the final deterministic ascending tie-break.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/ml/test_metric_registry.py tests/test_evaluator.py -q
git add swmm_resilience/ml/metric_registry.py tests/ml/test_metric_registry.py
git commit -m "feat: add configurable ML metric registry"
```

### Task 4: Persist Grouped OOF Evaluation And Selection

**Files:**
- Create: `swmm_resilience/ml/training.py`
- Create: `tests/ml/test_training_sqlite.py`
- Reuse: `swmm_resilience/database/training_queries.py`
- Reuse: `swmm_resilience/database/repositories.py`

**Interfaces:**
- Produces: `TrainingRequest`, `TrainingResult`, `train_and_select(conn, config, request) -> TrainingResult`, `rank_stored_candidates(conn, training_run_id, selection) -> CandidateRanking`
- Consumes: `training_samples_v17`, model registry, metric registry

- [ ] **Step 1: Write grouped-split and OOF persistence tests**

Seed at least six complete runs and assert:

```python
result = train_and_select(conn, config, TrainingRequest(target="inunda"))
assert result.contract_id == "tabular_v3_17"
assert set(result.fold_train_run_ids[0]).isdisjoint(result.fold_validation_run_ids[0])
assert scalar(conn, "SELECT COUNT(*) FROM oof_predictions") == expected_models * expected_samples
assert scalar(conn, "SELECT COUNT(*) FROM trained_models") == 1
```

Repeat for `vol_inundacion_m3`, verifying only flooded training rows enter the
oracle regression fit population, while the fitted regressor predicts every
validation row. Regression-only metrics use observed flooded validation rows;
the system pairing needs predictions on non-flooded rows to score classifier
false positives. Zero/negative predictions are reconciled as specified by the
current end-to-end rule.

Run `target="system"` and assert every classifier/regressor pairing is ranked
from aligned OOF rows, `total_volume_error_pct` uses classifier-gated volume,
the selected pair creates exactly two `trained_models` rows, and the two model
IDs share one training run. Add tests that changing the reporting/ranking
metric calls `rank_stored_candidates()` without adding model evaluations or
fitting estimators. Promoting a newly ranked winner is a separate final full-
data fit; it does not rerun folds. Changing any fitting parameter creates a new
`training_run_id` and new evaluations.

- [ ] **Step 2: Define request/result types**

```python
@dataclass(frozen=True)
class TrainingRequest:
    target: Literal["inunda", "vol_inundacion_m3", "system"] = "system"
    run_ids: tuple[int, ...] | None = None
    grouping: Literal["group_kfold", "loso"] = "group_kfold"
    loso_axis: Literal["scenario_key", "factor_mult", "shape_id"] = "scenario_key"


@dataclass(frozen=True)
class TrainingResult:
    training_run_id: int
    model_ids: tuple[int, ...]
    algorithms: tuple[str, ...]
    contract_id: str
    primary_metric: str
    primary_value: float
    fold_train_run_ids: tuple[tuple[int, ...], ...]
    fold_validation_run_ids: tuple[tuple[int, ...], ...]
```

- [ ] **Step 3: Implement one fold definition reused by all algorithms**

Build folds once and store their exact run IDs, then loop over algorithms.
`group_kfold` groups by `run_id`. `loso` groups by the requested persisted
`scenario_key`, `factor_mult`, or `shape_id`; reject null group values or fewer
than two distinct groups. Never ask each estimator to create its own CV.

For each target, algorithm, and fold:

1. slice by persisted row indexes;
2. create a fresh candidate;
3. fit fold-local preprocessing;
4. predict values and probabilities/scores;
5. insert one `model_evaluations` row and all OOF rows;
6. compute and insert metric rows;
7. mark that evaluation `COMPLETE`, or persist `FAILED` details and fail the
   enclosing training run.

For `target="system"`, align classification and regression OOF rows by
`(run_id, node_pk, fold_id)`. Evaluate the Cartesian product of classifier and
regressor algorithms. A predicted non-flooded node has end-to-end volume zero;
a predicted flooded node uses `max(0, regressor_prediction)`. Compute the
configured system primary metric and ordered tie-breakers on those aligned
rows. Do not fit another fold or overwrite target-specific OOF predictions.

- [ ] **Step 4: Rank and fit the final candidate**

Pool OOF predictions by algorithm, calculate configured primary/tie metrics,
and select deterministically. A target-specific request selects one algorithm;
a system request selects one classifier/regressor pair. Create fresh winning
candidates and fit all eligible data only after selection. Mark training status
`COMPLETE` only after Task 5 stores one target-specific artifact or both system
artifacts.

`rank_stored_candidates()` reads existing OOF rows and computes a ranking only;
it never calls `fit`. If the user explicitly promotes a different ranking
winner, refit only that final candidate (or pair) on all eligible rows and store
new artifacts with the new selection metric provenance.

- [ ] **Step 5: Verify and commit the OOF engine before artifact storage**

Use a temporary in-test artifact stub that records bytes, then run:

```powershell
python -m pytest tests/ml/test_training_sqlite.py -q
git add swmm_resilience/ml/training.py tests/ml/test_training_sqlite.py
git commit -m "feat: persist grouped OOF model evaluation"
```

### Task 5: Store And Verify Model BLOBs

**Files:**
- Create: `swmm_resilience/ml/artifacts.py`
- Create: `tests/ml/test_artifacts_sqlite.py`
- Modify: `swmm_resilience/ml/training.py`

**Interfaces:**
- Produces: `store_model() -> int`, `load_verified_model() -> VerifiedModel`
- Consumes: fitted sklearn-compatible estimator and manifest metadata

- [ ] **Step 1: Write BLOB safety tests**

Test round-trip equality, SHA mismatch, corrupt bytes, wrong target, wrong
contract ID, reversed feature order, and incompatible recorded major library
version. Also assert no `.joblib` file appears under `tmp_path`. For a system
training run, force the second BLOB insert to fail and assert neither selected
model row nor a `COMPLETE` training status remains.

- [ ] **Step 2: Implement in-memory serialization**

```python
def serialize_model(model) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()
```

Build a manifest with canonical JSON containing target, algorithm,
hyperparameters, contract ID/hash, exact ordered features, target transform,
query parameters, run IDs, sorted training network SHA-256 values, seed,
grouping, and Python/library versions.

- [ ] **Step 3: Verify before deserialization**

`load_verified_model()` must select by explicit `model_id`, verify BLOB hash
and all requested metadata, and only then call `joblib.load(io.BytesIO(blob))`.
Require the runtime Python major/minor and the exact recorded NumPy,
scikit-learn, XGBoost, and joblib versions; fail closed with an actionable
compatibility error before deserialization. Return:
Return:

```python
@dataclass(frozen=True)
class VerifiedModel:
    model_id: int
    target: str
    algorithm: str
    estimator: object
    manifest: dict[str, object]
```

- [ ] **Step 4: Integrate final storage transaction**

Store the selected model (or both selected system models) and mark
`training_runs.status='COMPLETE'` in one transaction. On
serialization/storage failure, roll back every artifact, then mark the
training run failed in a separate recovery transaction; do not leave a
selectable partial pair.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/ml/test_artifacts_sqlite.py tests/ml/test_training_sqlite.py -q
git add swmm_resilience/ml/artifacts.py swmm_resilience/ml/training.py tests/ml/test_artifacts_sqlite.py tests/ml/test_training_sqlite.py
git commit -m "feat: store verified model BLOBs in SQLite"
```

### Task 6: Implement SQLite-Backed Prediction

**Files:**
- Create: `swmm_resilience/ml/prediction.py`
- Create: `tests/ml/test_prediction_sqlite.py`
- Modify: `swmm_resilience/validation/hydrograph_batch.py:30-426`
- Modify: `main.py` prediction branches

**Interfaces:**
- Produces:
  - `predict_factor(conn, config, factor, classifier_id, regressor_id) -> pd.DataFrame`
  - `ScenarioPredictor.from_database(conn, ..., classifier_id, regressor_id)`
- Consumes: verified SQLite models and the shared v17 feature builder

- [ ] **Step 1: Write inference guardrail tests**

Cover valid prediction, wrong target IDs, 15-feature manifest, corrupt BLOB,
network mismatch, extrapolation flags, negative regressor output clipping, and
exact output columns:

```python
assert result.columns.tolist() == [
    "node_id", "inunda_pred", "prob_inunda", "vol_pred_m3", "extrapolated"
]
```

- [ ] **Step 2: Implement explicit model resolution**

For end-to-end prediction, require either both explicit IDs or neither. An
explicit classifier/regressor pair must belong to the same `COMPLETE`
`target='system'` training run and contract. If IDs are omitted, query complete
system training runs for the contract and configured system primary metric,
rank using registry direction/tie rules, break a remaining tie by ascending
`training_run_id`, and retrieve both target rows from the winner. Log/return
the resolved IDs; never select by filename or modification time.

Compare the inference network SHA-256 with the sorted hashes in each model
manifest. Default behavior rejects an unseen network. When the validated
prediction config explicitly enables network extrapolation, allow it and set
`extrapolated=True` on every returned row; never silently downgrade a mismatch
to a warning.

- [ ] **Step 3: Reuse one feature builder**

Both factor and hydrograph paths call the same public builder introduced in
Plan B, then:

```python
X = TABULAR_V3_17.validate_frame(frame.loc[:, FEATURE_COLUMNS_V17])
```

No `align_feature_columns()` behavior is permitted.

- [ ] **Step 4: Cut validation and root CLI to database model IDs**

Replace `--clf-path`/`--reg-path` with `--classifier-id`/`--regressor-id` and
default verified selection. Update hydrograph batch factory/tests to inject a
database connection or path and model IDs. Each validation scenario is a
persisted Plan B run; validation metrics are inserted into `model_metrics`
with `owner_kind='model'` and a scenario-specific `scope`. Plots consume the
returned/query DataFrames. Remove mandatory validation summary CSV writes;
`--export-csv PATH` remains the only explicit flat-file export boundary.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/ml/test_prediction_sqlite.py tests/test_scenario_predict.py tests/test_hydrograph_batch.py tests/test_simulate_single_factor.py -q
git add swmm_resilience/ml/prediction.py swmm_resilience/validation/hydrograph_batch.py main.py tests/ml/test_prediction_sqlite.py tests/test_scenario_predict.py tests/test_hydrograph_batch.py tests/test_simulate_single_factor.py
git commit -m "feat: predict from verified SQLite models"
```

### Task 7: Make Analysis Model-Family Aware

**Files:**
- Modify: `swmm_resilience/ml/feature_importance.py`
- Modify: `swmm_resilience/ml/feature_analysis.py`
- Create: `tests/ml/test_model_explanations.py`
- Modify: existing feature-analysis tests

**Interfaces:**
- Produces: `explain_model(verified_model, X, output_dir)`
- Consumes: linear, tree, or SVM verified models

- [ ] **Step 1: Write family-routing tests**

Assert linear models use coefficients, tree models use supported native/SHAP
paths, and SVM uses permutation importance. Assert every output reports raw
feature names from `FEATURE_COLUMNS_V17` and readable labels.

- [ ] **Step 2: Implement capability-based routing**

Do not route solely by string algorithm names. Inspect the final estimator
inside the pipeline/target wrapper and select a supported explainer. Raise a
clear error for an unknown estimator instead of returning an empty plot.

- [ ] **Step 3: Read data/models from SQLite**

Analysis accepts `model_id` and optional run IDs, loads the verified model and
training view, and never reads `classifier.joblib` or `regressor.joblib`.
Replace every feature-analysis test import of `trainer.FEATURE_COLS` with
`contracts.FEATURE_COLUMNS_V17` so Plan D can delete the compatibility alias.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/ml/test_model_explanations.py tests/ml/test_feature_analysis.py tests/ml/test_feature_importance_labels.py -q
git add swmm_resilience/ml/feature_importance.py swmm_resilience/ml/feature_analysis.py tests/ml/test_model_explanations.py tests/ml/test_feature_analysis.py tests/ml/test_feature_importance_labels.py
git commit -m "feat: explain every supported model family"
```

### Task 8: Run The Unified ML Phase Gate

**Files:**
- Verify only

**Interfaces:**
- Produces: approved replacements required for Plan D deletions
- Consumes: Tasks 1-7

- [ ] **Step 1: Run the fast ML safety suite**

```powershell
python -m pytest tests/ml/test_feature_contract_v17.py tests/ml/test_model_registry.py tests/ml/test_metric_registry.py tests/ml/test_training_sqlite.py tests/ml/test_artifacts_sqlite.py tests/ml/test_prediction_sqlite.py tests/ml/test_model_explanations.py -q
```

- [ ] **Step 2: Run current evaluation and validation regressions**

```powershell
python -m pytest tests/test_evaluator.py tests/test_scenario_predict.py tests/test_hydrograph_batch.py tests/test_factor_comparison_cli.py tests/test_resilience.py tests/test_flood_volume.py -q
```

- [ ] **Step 3: Prove flat files are absent from new ML code**

```powershell
rg -n "joblib\.load|joblib\.dump|classifier\.joblib|regressor\.joblib|dataset_final|dataset_ml|read_csv" swmm_resilience/ml/training.py swmm_resilience/ml/artifacts.py swmm_resilience/ml/prediction.py
```

Expected: `joblib.dump/load` appears only against `BytesIO` in `artifacts.py`;
no filesystem paths or CSV reads appear.

- [ ] **Step 4: Check branch cleanliness**

```powershell
git diff --check
git status --short --branch
```

Expected: no uncommitted implementation files.
