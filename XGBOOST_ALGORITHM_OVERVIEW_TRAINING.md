# XGBoost Algorithm Overview, Hyperparameter Tuning, and Training

This note describes the current XGBoost architecture used by the active project
pipeline. It is written for presentation preparation, so it focuses on what the
system does, what data enters the models, how training works, how tuning is
handled, and which code paths should not be confused with the current
implementation.

## 1. Executive Summary

The active project is a SWMM-to-ML surrogate pipeline for the Chico Sur sewer
network. SWMM is still the hydraulic source of truth: it generates simulated
flooding results for multiple inflow factors. The ML layer learns from those
simulations so that later it can estimate flooded nodes and flood volume much
faster than rerunning SWMM for every scenario.

The active XGBoost architecture is a two-stage tabular surrogate:

1. **Classification stage:** predict whether each junction floods.
2. **Regression stage:** for nodes predicted as flooded, estimate the flood
   volume in cubic meters.

In code, the active production-style training path is:

```text
main.py
  -> swmm_resilience.dataset.assembler.assemble_dataset()
  -> swmm_resilience.ml.trainer.train_models()
  -> swmm_resilience.ml.evaluator.evaluate_models()
  -> swmm_resilience.ml.feature_importance.generate_feature_importance_plots()
```

The main active ML files are:

- `swmm_resilience/ml/trainer.py`
- `swmm_resilience/ml/evaluator.py`
- `swmm_resilience/ml/predict.py`
- `swmm_resilience/ml/feature_importance.py`
- `config.yaml`

The current dataset is `data/training/dataset_final.csv`. At the time of this
review, it contains 4,000 rows, 160 unique junction nodes, and 25 factors from
0.20 to 5.00. The label balance is 2,813 non-flooded rows and 1,187 flooded
rows.

## 2. Scope and Legacy-Code Warning

There are multiple ML paths in the repository, and not all of them represent
the current presentation story.

### Current active path

The active spec v4 path is the CLI pipeline in root `main.py`. The README says
this version is a CLI pipeline based on CSV files, `joblib` models, JSON
metrics, and PNG maps. It also says the spec v4 flow does not use the desktop
frontend, SQLite, or temporal legacy modules as the main path.

For the XGBoost architecture presentation, the safest source of truth is:

- `main.py`
- `config.yaml`
- `swmm_resilience/ml/trainer.py`
- `swmm_resilience/ml/evaluator.py`
- `swmm_resilience/ml/predict.py`
- `swmm_resilience/extraction/*`
- `swmm_resilience/dataset/assembler.py`

### Do not present these as the active XGBoost training architecture

The following are still in the repository but should be treated carefully:

- `swmm_resilience/ml/train.py`: older model-comparison and artifact path. It
  compares Ridge, Lasso, SVR, and XGBoost with PCA and grouped train/test
  splits. Useful historically, but root `main.py` currently imports
  `swmm_resilience.ml.trainer.train_models`, not this module.
- `swmm_resilience/ml/predict_tabular.py`: CSV inference compatibility path.
  README labels CSV prediction as legacy.
- `swmm_resilience/ml/predict_from_inp.py` and `swmm_resilience/ml/scenario_predict.py`:
  related to Pipeline A / prior inference workflows. They may still be wired to
  tests or UI code, so they are not automatically trash, but they are not the
  current spec v4 training story.
- `swmm_resilience/ml/temporal/*`: temporal CNN/LSTM work and surrogate
  experiments. Important for future work, not the current XGBoost architecture.
- `pruebas_locales.py`, `verificar.py`, and `halve_timeseries.py`: the audit
  document marks these as scratch or risky one-off scripts. They should not be
  used for presentation content.

## 3. End-to-End Algorithm Overview

At a high level, the algorithm converts hydraulic simulations into a supervised
learning table:

```text
SWMM .inp network
  -> extract static node and topology features
  -> run SWMM for each inflow factor
  -> parse flood volumes from SWMM .rpt files
  -> assemble one row per node per factor
  -> train XGBoost classifier and XGBoost regressor
  -> evaluate with grouped factor-based validation
  -> save models and use them for fast inference
```

Each row of the training dataset represents:

```text
one junction node under one simulated inflow factor
```

The model does not predict a whole network with one row. It predicts node-level
behavior, then network-level summaries are built by aggregating node
predictions.

## 4. Data Generation and Dataset Assembly

### 4.1 Static features

Static features are extracted from the SWMM `.inp` file by
`swmm_resilience/extraction/static_features.py`. These features describe the
physical and local hydraulic context of each junction.

Examples:

- `elev_fondo`: invert elevation / node bottom elevation.
- `prof_max`: maximum node depth.
- `n_tuberias_in`: number of incoming pipes.
- `diam_max_in`: maximum upstream pipe diameter.
- `diam_max_out`: maximum downstream pipe diameter.
- `pendiente_max_in`: maximum incoming slope.
- `pendiente_out`: outgoing pipe slope.
- `base_inflow_lps`: peak base inflow from the embedded SWMM inflow or
  timeseries.
- `coord_x`, `coord_y`: coordinates used for maps, not model inputs.

### 4.2 Topology features

Topology features are added by `swmm_resilience/extraction/topology.py` using a
directed graph of the sewer network.

Examples:

- `dist_outfall_m`: shortest distance from the node to an outfall.
- `n_nodos_aguas_arriba`: number of upstream nodes draining into this node.
- `q_pico_acum_base`: accumulated base peak inflow from the node and all
  upstream ancestors.
- `upstream_capacity_lps`: estimated full-flow capacity of immediate upstream
  conduits.

These variables help XGBoost learn not only from a node's own attributes, but
also from its position in the drainage network.

### 4.3 Dynamic scenario features

Dynamic features are calculated in
`swmm_resilience/extraction/dynamic_features.py`.

For the training pipeline, every scenario is a uniform scaling of the base
hydrographs:

```text
q_pico_nodo = base_inflow_lps * factor
q_pico_acum_escalado = q_pico_acum_base * factor
```

The dataset also stores `factor_mult`, but the active feature contract
explicitly excludes it from the model. This is important: `factor_mult` is a
global scenario descriptor, but it does not have a valid meaning for arbitrary
hydrographs. The model receives the physically meaningful node-level and
upstream peak quantities instead.

### 4.4 Labels

Labels come from SWMM report files through
`swmm_resilience/extraction/labels.py`.

The canonical label rule is:

```text
inunda = 1 if vol_inundacion_m3 >= flood_threshold_m3
```

The configured threshold in `config.yaml` is:

```yaml
dataset:
  flood_threshold_m3: 1.0
```

Nodes absent from the SWMM flooding summary are assigned zero volume and
`inunda = 0`.

### 4.5 Final assembled dataset

`swmm_resilience/dataset/assembler.py` joins:

- static and topology features,
- dynamic features for each factor,
- SWMM-derived labels.

It writes the final CSV to:

```text
data/training/dataset_final.csv
```

The current dataset columns are:

```text
node_id
elev_fondo
prof_max
n_tuberias_in
n_tuberias_out
diam_max_in
diam_max_out
pendiente_max_in
pendiente_out
base_inflow_lps
dist_outfall_m
n_nodos_aguas_arriba
q_pico_acum_base
upstream_capacity_lps
coord_x
coord_y
factor_mult
q_pico_nodo
q_pico_acum_escalado
vol_inundacion_m3
inunda
```

## 5. Active Feature Contract

The model input columns are fixed in `swmm_resilience/ml/trainer.py` as
`FEATURE_COLS`:

```text
elev_fondo
prof_max
n_tuberias_in
diam_max_in
diam_max_out
pendiente_max_in
pendiente_out
base_inflow_lps
dist_outfall_m
n_nodos_aguas_arriba
q_pico_acum_base
upstream_capacity_lps
q_pico_nodo
q_pico_acum_escalado
```

Important exclusions:

- `node_id` is excluded to avoid memorizing node names.
- `coord_x` and `coord_y` are excluded from the model and used for maps.
- `factor_mult` is excluded from the model, but retained as metadata.
- `vol_inundacion_m3` is the regression target, not an input.
- `inunda` is the classification target, not an input.

This feature contract is also protected by tests. The prediction test asserts
that both classifier and regressor receive columns in exactly the same order as
`FEATURE_COLS`.

## 6. XGBoost Architecture

The active architecture is not a single model. It is a two-model system:

```text
X features
  -> XGBClassifier
      -> inunda_pred
          if inunda_pred = 0: predicted volume = 0
          if inunda_pred = 1: XGBRegressor predicts log-volume
              -> expm1(log prediction)
              -> clip negative values to 0
```

### 6.1 Classifier

The classifier predicts:

```text
inunda
```

This is a binary target:

- `0`: node does not flood.
- `1`: node floods above the configured threshold.

When `config.yaml` sets the classifier algorithm to `xgboost`, the code builds
an `XGBClassifier`.

Pipeline:

```text
SimpleImputer(strategy="median")
  -> XGBClassifier
```

The imputer handles missing values in hydraulic/topological features such as
upstream diameter or slope for headwater nodes.

The classifier uses `eval_metric="logloss"` and `random_state=42`.

### 6.2 Class imbalance handling

Flooding is not perfectly balanced. In the current dataset:

```text
non-flooded rows: 2813
flooded rows:     1187
```

The classifier uses `scale_pos_weight`.

In `config.yaml`:

```yaml
scale_pos_weight: "auto"
```

In training, `"auto"` becomes:

```text
scale_pos_weight = number_of_non_flooded_rows / number_of_flooded_rows
```

With the current dataset this is approximately:

```text
2813 / 1187 = 2.37
```

That tells XGBoost to give extra weight to the minority flooded class.

### 6.3 Regressor

The regressor predicts:

```text
vol_inundacion_m3
```

However, it is trained only on rows where:

```text
inunda = 1
```

This is intentional. Non-flooded nodes have zero flood volume, and including a
large number of zeros in the regressor would make it learn the classification
problem again instead of learning flood severity. The classifier decides
whether the node floods; the regressor estimates severity conditional on
flooding.

Pipeline:

```text
SimpleImputer(strategy="median")
  -> XGBRegressor
```

The target is transformed before fitting:

```text
y_reg_train = log1p(vol_inundacion_m3)
```

At inference time, predictions are transformed back:

```text
vol_pred_m3 = expm1(predicted_log_volume)
```

Negative reconstructed volumes are clipped to zero.

This log-space training is tested in `tests/test_ml_trainer_predict.py`.

## 7. Hyperparameter Tuning

### 7.1 What tuning exists today

The current active pipeline does **not** implement automated grid search,
random search, Bayesian optimization, or Optuna.

Instead, tuning is configuration-driven and evaluation-driven:

1. Hyperparameters are declared in `config.yaml`.
2. Models are trained with those values.
3. Evaluation is run with LOSO and/or GroupKFold5.
4. Metrics are written to JSON.
5. The user adjusts `config.yaml` and reruns the pipeline if a new
   configuration should be tested.

This is important for the presentation: the project has configurable
hyperparameters, but not a built-in automatic hyperparameter optimization loop
in the active spec v4 path.

### 7.2 Current XGBoost hyperparameters

From `config.yaml`, the active classifier settings are:

```yaml
classifier:
  algorithm: "xgboost"
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  scale_pos_weight: "auto"
```

The active regressor settings are:

```yaml
regressor:
  algorithm: "xgboost"
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
```

The implementation passes these into `XGBClassifier` and `XGBRegressor` in
`swmm_resilience/ml/trainer.py`.

### 7.3 Meaning of each hyperparameter

`n_estimators = 200`

Number of boosting trees. More trees can model more complex relationships, but
also increase training time and overfitting risk. Here, 200 is a moderate
setting for a small-to-medium tabular dataset.

`max_depth = 6`

Maximum depth of each tree. Deeper trees learn more interactions between
hydraulic/topological variables, such as combinations of upstream capacity,
local depth, and accumulated peak flow. Depth 6 allows nonlinear interactions
without making individual trees extremely deep.

`learning_rate = 0.05`

Shrinkage applied to each tree's contribution. A smaller learning rate makes
boosting more gradual and usually more stable, especially with many trees.

`subsample = 0.8`

Each tree trains on 80% of the rows. This introduces randomness and can reduce
overfitting.

`scale_pos_weight = "auto"`

Classifier-only. It compensates for class imbalance by weighting positive
flooding examples more heavily.

`random_state = 42`

Set in code for reproducibility.

`eval_metric = "logloss"`

Classifier-only. It tells XGBoost to evaluate classification learning with
logarithmic loss internally.

### 7.4 Parameters not currently tuned in the active path

The active `trainer.py` does not expose every common XGBoost parameter. For
example, the active path does not currently configure:

- `colsample_bytree`
- `min_child_weight`
- `gamma`
- `reg_alpha`
- `reg_lambda`
- `early_stopping_rounds`
- validation sets for early stopping

Older `swmm_resilience/ml/train.py` contains a separate `ML_MODEL_CONFIGS`
dictionary with `colsample_bytree`, PCA, and model comparison. That should not
be presented as the current root `main.py` training behavior unless the
presentation explicitly discusses older/alternative code.

## 8. Training Procedure

### 8.1 Full training run

The normal full pipeline is:

```bash
python main.py
```

If the dataset already exists and only ML should run:

```bash
python main.py --only-ml
```

Both paths eventually call:

```python
train_models(df, config, MODELS_DIR)
```

where:

```text
MODELS_DIR = outputs/models
```

### 8.2 Training steps inside `train_models`

The function does the following:

1. Creates the output directory.
2. Selects `X = df[FEATURE_COLS]`.
3. Selects classifier target `y_clf = df["inunda"]`.
4. Computes `scale_pos_weight` from the class balance.
5. Builds the classifier pipeline.
6. Fits the classifier on all rows.
7. Filters the dataset to flooded rows only.
8. Builds the regressor pipeline.
9. Fits the regressor on flooded rows using `log1p(vol_inundacion_m3)`.
10. Saves:
    - `outputs/models/classifier.joblib`
    - `outputs/models/regressor.joblib`
    - `outputs/models/training_inp_hash.txt`

The saved hash records the MD5 of the training `.inp` file. Prediction checks
this hash before inference so the model is not silently used on a changed
network.

### 8.3 Why the final models train on the full dataset

`train_models` trains the saved final artifacts on the full available dataset.
Evaluation is handled separately by `evaluate_models`, which repeatedly trains
fresh fold models under LOSO and GroupKFold5. This separation is normal:

- evaluation estimates out-of-sample behavior;
- final artifacts use all data available after evaluation.

## 9. Evaluation Strategy

Evaluation is implemented in `swmm_resilience/ml/evaluator.py`.

Configured methods:

```yaml
evaluation:
  methods:
    - "LOSO"
    - "GroupKFold5"
  stratify_by_factor: true
```

### 9.1 Grouping variable

The active evaluator groups by:

```text
factor_mult
```

This means validation folds hold out complete inflow factors. That is why the
summary printed by `main.py` calls the primary result LOSO: Leave One Scenario
Out / Leave One factor Out in practical terms.

### 9.2 Three evaluation levels

The evaluator reports three levels:

#### Level 1: classifier

Measures only the binary flooding decision.

Metrics:

- precision
- recall
- F1
- ROC AUC, where both classes are present

#### Level 2: regressor oracle

Measures the regressor only on truly flooded nodes. This uses the real label to
select flooded test rows, so it is an optimistic upper bound for volume
prediction conditional on knowing which nodes flood.

Metrics:

- NSE
- log-NSE
- RMSE
- MAE
- R2

`log_nse` is computed in log space. Tests verify that this is not the same as
ordinary NSE and that pooled predictions are used before calculating NSE.

#### Level 3: end-to-end system

Measures the real two-stage system:

1. classifier predicts flooded nodes;
2. regressor predicts volume only for predicted flooded nodes;
3. all other nodes get zero predicted volume.

Metrics:

- percentage of correctly classified nodes;
- RMSE over all nodes;
- predicted total flood volume;
- real total flood volume.

### 9.3 Current metric artifacts

The current metrics in `outputs/metrics` are:

Classifier, primary LOSO:

```json
{
  "precision": 0.7290,
  "recall": 0.7730,
  "f1": 0.7417,
  "auc_roc": 0.9959
}
```

Regressor oracle:

```json
{
  "nse": 0.9903,
  "log_nse": 0.9441,
  "rmse": 16.0638,
  "mae": 10.3864,
  "r2": 0.9903
}
```

End-to-end:

```json
{
  "pct_nodos_correctos": 0.9805,
  "rmse_vol_todos_nodos": 7.5200,
  "vol_total_pred_m3": 6890.5841,
  "vol_total_real_m3": 7030.56
}
```

Interpretation for presentation:

- The classifier has very high ranking ability by AUC, but its primary LOSO F1
  is lower because the decision threshold and low-factor scenarios make the
  positive class difficult.
- The oracle regressor is very strong when the true flooded nodes are known.
- The end-to-end total network volume is close to SWMM in the current metric
  artifact, with predicted total volume lower than real total volume.

## 10. Inference Procedure

Inference is implemented in `swmm_resilience/ml/predict.py`.

Given a factor:

```bash
python main.py --predict --factor 3.5
```

The system:

1. Loads `classifier.joblib` and `regressor.joblib`.
2. Reads `training_inp_hash.txt`.
3. Computes the current `.inp` MD5.
4. Aborts if the current `.inp` differs from the training `.inp`.
5. Re-extracts static and topology features from the `.inp`.
6. Computes dynamic features for the requested factor.
7. Runs classifier prediction.
8. Runs regressor prediction only where `inunda_pred = 1`.
9. Returns:
   - `node_id`
   - `inunda_pred`
   - `vol_pred_m3`
   - `coord_x`
   - `coord_y`

The same predictor is used by maps and factor-comparison plots.

## 11. Feature Importance

After training, `main.py` calls:

```python
generate_feature_importance_plots(clf, reg, METRICS_DIR)
```

This creates:

```text
outputs/metrics/feature_importance_classifier.png
outputs/metrics/feature_importance_regressor.png
```

The implementation reads `model.feature_importances_` from the XGBoost model
inside each pipeline and plots the values against human-readable feature names.

This gives a presentation-friendly way to discuss which hydraulic variables the
classifier and regressor rely on most.

## 12. What to Say in the Presentation

A concise presentation framing:

> The current surrogate model is a two-stage XGBoost system. First, an
> XGBClassifier decides whether each junction floods. Then, for nodes classified
> as flooded, an XGBRegressor estimates flood volume in cubic meters. The
> regressor is trained only on flooded rows and uses a log1p target transform to
> stabilize volume prediction. The models use static network features,
> topology features, and scenario peak-flow features, while excluding node IDs,
> coordinates, raw factor metadata, and target columns.

For hyperparameter tuning:

> Hyperparameters are currently controlled in `config.yaml`, not by an
> automatic search routine. The selected configuration uses 200 trees, depth 6,
> learning rate 0.05, and 80% row subsampling. The classifier additionally uses
> automatic positive-class weighting to compensate for the imbalance between
> flooded and non-flooded rows. Model quality is checked with factor-grouped
> evaluation using LOSO and GroupKFold5.

For training:

> Training uses the full assembled dataset for final saved artifacts. Evaluation
> is separate and retrains fold-specific models to estimate generalization
> across inflow factors. The saved artifacts are `classifier.joblib`,
> `regressor.joblib`, and a training `.inp` hash that prevents accidental
> inference with a changed network.

## 13. Limitations and Risks

1. **No automated hyperparameter optimization yet.** Tuning is manual through
   `config.yaml`.
2. **Validation is grouped by factor, not by network.** This is appropriate for
   the current single-network dataset, but it does not prove generalization to
   other sewer networks.
3. **The active XGBoost path excludes `factor_mult`.** This is deliberate, but
   it means the dynamic signal must be represented correctly by `q_pico_nodo`
   and `q_pico_acum_escalado`.
4. **The regressor depends on classifier routing.** If the classifier misses a
   flooded node, the end-to-end system assigns zero volume even if the regressor
   could have estimated a positive value.
5. **Older code can confuse the story.** `ml/train.py` includes PCA and
   multi-model comparison, while `trainer.py` is the active root pipeline. Keep
   these separate in the presentation.

## 14. Source Map

Use these files when checking or extending the presentation:

- `main.py`: CLI orchestration and active pipeline.
- `config.yaml`: active XGBoost and evaluation hyperparameters.
- `swmm_resilience/ml/trainer.py`: active XGBoost model construction and final
  artifact training.
- `swmm_resilience/ml/evaluator.py`: LOSO / GroupKFold5 evaluation and JSON
  metric output.
- `swmm_resilience/ml/predict.py`: inference flow and `.inp` hash guard.
- `swmm_resilience/ml/feature_importance.py`: feature-importance plots.
- `swmm_resilience/extraction/static_features.py`: static node features.
- `swmm_resilience/extraction/topology.py`: graph/topology features.
- `swmm_resilience/extraction/dynamic_features.py`: factor and hydrograph peak
  features.
- `swmm_resilience/extraction/labels.py`: canonical flood-volume label rule.
- `swmm_resilience/dataset/assembler.py`: final dataset assembly.
- `tests/test_ml_trainer_predict.py`: tests for artifacts, log-space regressor
  training, and prediction feature order.
- `tests/test_evaluator.py`: tests for evaluator metric behavior and JSON
  outputs.
- `AUDITORIA_CODIGO_BASURA_2026-06-15.md`: audit notes on scratch scripts and
  Pipeline A caution.
