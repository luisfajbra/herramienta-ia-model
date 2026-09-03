# SQLite V17 Pipeline Consolidation Design

> **SUPERSEDED (2026-09-03):** this full-consolidation vision (SimulationStore,
> unified ML registry, legacy retirement) was never executed — none of Plans
> A-D's checkboxes are checked. The branch instead shipped a much smaller
> "minimal path" (`csv_backfill.py`), and the active plan going forward is
> `docs/FLUJO_ACTUAL.md` §12 (only the 24-column training frame moves to
> SQL; CSV/`.joblib`/legacy GUI stay). Kept for historical reference only —
> do not resume this roadmap without re-confirming with the project owner.

**Status:** Approved in conversation; ready for implementation planning
**Date:** 2026-08-21
**Branch:** `cleanup/sqlite-v17-pipeline-consolidation`
**Base:** `main` at `af9873a`

## 1. Purpose

Consolidate the repository around one operational SWMM-to-ML pipeline with an
explicit 17-feature tabular contract and one local SQLite database as the
source of truth. Preserve the useful model families from the older tabular
pipeline, remove the duplicate legacy architecture only after those
capabilities have replacements, and leave CNN/LSTM code in a clearly defined
pending state for a later redesign against the new database.

The cleanup is deliberately conservative. A file is not removed merely
because it looks old. Its useful behavior must first be migrated or proven
unneeded, its consumers must be identified, and the relevant tests must pass
before and after deletion.

## 2. Approved Outcomes

The completed repository will have:

- one root CLI entry point: `main.py`;
- one structured configuration source: `config.yaml`;
- one local database by default: `data/swmm_resilience.sqlite3`;
- one tabular input contract: `tabular_v3_17`;
- one training/evaluation implementation shared by all tabular algorithms;
- SQLite-backed simulation inputs, outputs, temporal observations, training
  provenance, out-of-fold predictions, metrics, and serialized models;
- no mandatory `dataset_final.csv` or `dataset_ml.csv` intermediate;
- no versioned `.joblib`, `.pkl`, `.pt`, `.pth`, `.h5`, or `.keras` model
  artifacts;
- no operational desktop, duplicate orchestration, or duplicate tabular ML
  pipeline;
- concise current documentation plus a short retired-architecture record;
- CNN/LSTM source retained but unavailable through operational commands until
  it is redesigned against the new schema.

## 3. Non-Goals

This consolidation does not:

- retrain or preserve any 15-feature model;
- import historical model binaries into the new database;
- make SQLite a multi-user or multi-writer service;
- fully redesign, train, or deploy CNN/LSTM models;
- add hyperparameter optimization as part of the cleanup;
- add a new desktop or web UI;
- delete the user's untracked presentations, thesis document, or other local
  work;
- delete local generated datasets or results without a separate explicit user
  action.

## 4. Evidence And Starting State

Repository history identifies commit `56a7115` as the change that expanded
`FEATURE_COLS` from 15 to 17 by adding `duracion_horas` and
`tiempo_al_pico_h`. Current code already has a 17-column contract, while
README and historical documents still describe the 15-feature generation.

The repository currently contains two overlapping tabular architectures:

1. The root `main.py` spec-v4 flow using `config.yaml`,
   `swmm_resilience/ml/trainer.py`, and the 17 features.
2. The older package entry point, desktop, SQLite schema, dynamic feature
   discovery, PCA-oriented training, CSV prediction, and filesystem artifact
   flow.

The older flow is not technically orphaned: it still has imports and tests.
Its removal is therefore a controlled migration, not a blind dead-code
deletion.

The isolated worktree baseline on Python 3.11.9 is:

- 301 tests collected;
- 301 passed;
- 0 failed;
- runtime approximately 134 seconds;
- existing warnings include temporal inference without manifests and library
  deprecations.

Dependency installation has one pre-existing incompatibility:
`requirements.txt` pins `shap==0.52.0`, which requires Python 3.12, while this
project baseline uses Python 3.11. The installed SHAP 0.51.0 passes the current
suite. The implementation plan must resolve the Python support policy and pin
a compatible SHAP version before depending on a clean install.

## 5. Target Architecture

```text
config.yaml + SWMM .inp / hydrograph input
  -> simulation and validation
  -> static, topology, dynamic, and temporal extraction
  -> SQLite source of truth
       -> training_samples_v17 view
       -> grouped evaluation and OOF predictions
       -> selected fitted model BLOBs and manifests
       -> inference and hydrograph validation
       -> metrics and visualizations
```

The root CLI is the only supported coordinator. It invokes focused modules
whose public boundaries are:

- `database`: schema migration, connection policy, repositories, and queries;
- `simulation`: SWMM execution and raw hydraulic extraction;
- `extraction`: construction of the exact approved features and targets;
- `ml.contracts`: feature and target contracts;
- `ml.models`: algorithm registry and preprocessing rules;
- `ml.training`: grouped fitting, OOF generation, final fit, and persistence;
- `ml.metrics`: metric registry, calculation, ranking direction, and ties;
- `ml.artifacts`: safe model serialization, hashing, storage, and loading;
- `ml.prediction`: contract-checked factor and hydrograph inference;
- `analysis` and `visualization`: SQL-backed reporting and plots.

Files should be organized by these responsibilities during implementation.
The plan may choose exact filenames after mapping existing reusable functions,
but it must not recreate parallel old/new implementations.

## 6. SQLite Operating Contract

SQLite is appropriate because the application is local and has one user and
one writer. SQLite supports many readers but only one simultaneous writer, so
the application will centralize writes through one repository/connection
boundary. See the official SQLite documentation for
[transactions](https://www.sqlite.org/lang_transaction.html),
[WAL](https://www.sqlite.org/wal.html),
[limits](https://www.sqlite.org/limits.html), and
[`PRAGMA optimize`](https://www.sqlite.org/lang_analyze.html).

Connection policy:

- `PRAGMA foreign_keys = ON` on every connection;
- `PRAGMA journal_mode = WAL` during database initialization, with the returned
  mode checked rather than assumed;
- a finite busy timeout;
- explicit transactions for every logical write unit;
- batched `executemany` inserts for node and timestep rows;
- no commit per row or per timestep;
- `PRAGMA optimize` after schema/index changes and at orderly close after
  substantial writes;
- explicit WAL checkpoint before making a standalone backup or copy;
- database path configured under `database.path`, defaulting to
  `data/swmm_resilience.sqlite3`.

The database file, `-wal`, and `-shm` files are runtime data and must be ignored
by Git.

## 7. Database Schema

All tables use foreign keys. Integer surrogate keys are used for high-volume
joins; natural identities are protected by unique constraints. Timestamps are
stored as UTC ISO-8601 text. JSON fields use canonical UTF-8 JSON and are
validated by application code.

### 7.1 Schema and source identity

`schema_migrations`

- `version INTEGER PRIMARY KEY`
- `name TEXT NOT NULL UNIQUE`
- `checksum_sha256 TEXT NOT NULL`
- `applied_at_utc TEXT NOT NULL`

`networks`

- `network_id INTEGER PRIMARY KEY`
- `network_sha256 TEXT NOT NULL UNIQUE`
- `name TEXT NOT NULL`
- `source_filename TEXT NOT NULL`
- `inp_bytes BLOB NOT NULL`
- `flow_units TEXT NOT NULL`
- `created_at_utc TEXT NOT NULL`

The original `.inp` bytes are stored so the database remains self-contained.
Every use verifies `network_sha256` against those bytes.

`nodes`

- `node_pk INTEGER PRIMARY KEY`
- `network_id INTEGER NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE`
- `node_id TEXT NOT NULL`
- coordinates and stable node attributes needed for maps and extraction
- `UNIQUE(network_id, node_id)`

`links`

- `link_pk INTEGER PRIMARY KEY`
- `network_id INTEGER NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE`
- `link_id TEXT NOT NULL`
- endpoint node keys and stable link attributes
- `UNIQUE(network_id, link_id)`

### 7.2 Scenario and simulation lifecycle

`scenarios`

- `scenario_id INTEGER PRIMARY KEY`
- `network_id INTEGER NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE`
- `scenario_key TEXT NOT NULL`
- `scenario_kind TEXT NOT NULL`
- factor, shape identity, duration, and time-to-peak metadata where applicable
- canonical scenario/config JSON and SHA-256
- `UNIQUE(network_id, scenario_key, config_sha256)`

`scenario_inflows`

- `scenario_id INTEGER NOT NULL REFERENCES scenarios(scenario_id) ON DELETE CASCADE`
- `node_pk INTEGER NOT NULL REFERENCES nodes(node_pk) ON DELETE CASCADE`
- `step_index INTEGER NOT NULL`
- `time_sec REAL NOT NULL CHECK(time_sec >= 0)`
- `inflow_lps REAL NOT NULL CHECK(inflow_lps >= 0)`
- `PRIMARY KEY(scenario_id, node_pk, step_index)`

`runs`

- `run_id INTEGER PRIMARY KEY`
- `scenario_id INTEGER NOT NULL REFERENCES scenarios(scenario_id)`
- `status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETE','FAILED'))`
- start/end timestamps, SWMM version, configuration hash, continuity error,
  failure stage/type/message, and row-count checks

Only `COMPLETE` runs are eligible for training or model evaluation.

### 7.3 Exact tabular features and targets

`node_features`

- `run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE`
- `node_pk INTEGER NOT NULL REFERENCES nodes(node_pk) ON DELETE CASCADE`
- the 17 numeric feature columns defined in Section 8; physically undefined
  static/topology values use SQL `NULL` only for the contract-approved
  nullable features
- `feature_contract_id TEXT NOT NULL CHECK(feature_contract_id = 'tabular_v3_17')`
- `PRIMARY KEY(run_id, node_pk)`

`node_results`

- `run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE`
- `node_pk INTEGER NOT NULL REFERENCES nodes(node_pk) ON DELETE CASCADE`
- `inunda INTEGER NOT NULL CHECK(inunda IN (0,1))`
- `vol_inundacion_m3 REAL NOT NULL CHECK(vol_inundacion_m3 >= 0)`
- other retained aggregate hydraulic outputs with explicit units
- `PRIMARY KEY(run_id, node_pk)`

`training_samples_v17` is a view joining `runs`, `scenarios`,
`node_features`, and `node_results`. It exposes complete rows only and places
the 17 feature columns in canonical order. Model code may not construct an
alternative input set by selecting all numeric columns.

### 7.4 Temporal source data

`node_timeseries`

- `run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE`
- `node_pk INTEGER NOT NULL REFERENCES nodes(node_pk) ON DELETE CASCADE`
- `step_index INTEGER NOT NULL`
- `time_sec REAL NOT NULL CHECK(time_sec >= 0)`
- `total_inflow_lps REAL NOT NULL`
- `lateral_inflow_lps REAL NOT NULL`
- `depth_m REAL NOT NULL`
- `depth_ratio REAL NOT NULL`
- `flooding_lps REAL NOT NULL`
- `total_outflow_lps REAL NOT NULL`
- `failed_now INTEGER NOT NULL CHECK(failed_now IN (0,1))`
- `PRIMARY KEY(run_id, node_pk, step_index)`

The implementation plan must benchmark whether this high-volume composite key
benefits from `WITHOUT ROWID`. It may be used only if measured storage and
query performance improve; the SQLite documentation recommends measuring this
optimization rather than assuming it is beneficial.

### 7.5 Training, predictions, metrics, and model BLOBs

`training_runs`

- `training_run_id INTEGER PRIMARY KEY`
- contract ID/hash, target, canonical SQL query and parameters, exact included
  run IDs, grouping strategy, folds, random seed, library versions, timestamps,
  status, and failure details

`model_evaluations`

- model family, algorithm, hyperparameter JSON, fold ID, train/validation run
  IDs, status, and timing for each evaluated candidate

`oof_predictions`

- evaluation ID, run ID, node key, target, observed value, predicted value,
  class probability when applicable, and fold ID
- uniqueness prevents more than one OOF prediction per evaluation/sample

`model_metrics`

- owner kind/ID, scope, metric name, numeric value, validity flag, and reason
  when undefined

`trained_models`

- `model_id INTEGER PRIMARY KEY`
- training run, target, selected algorithm, preprocessing description,
  hyperparameters, contract ID/hash, ordered feature JSON, target transform,
  library versions, created timestamp, model SHA-256, and serialized model BLOB

The BLOB hash is verified before deserialization. Loading fails if the feature
contract, target, ordered columns, model hash, or required library compatibility
does not match. No legacy fallback or automatic column filling is allowed.

## 8. Feature Contract

The sole tabular contract ID is `tabular_v3_17`. Its ordered inputs are:

1. `elev_fondo`
2. `prof_max`
3. `n_tuberias_in`
4. `n_tuberias_out`
5. `diam_max_in`
6. `diam_max_out`
7. `pendiente_max_in`
8. `pendiente_out`
9. `base_inflow_lps`
10. `dist_outfall_m`
11. `n_nodos_aguas_arriba`
12. `q_pico_acum_base`
13. `upstream_capacity_lps`
14. `q_pico_nodo`
15. `q_pico_acum_escalado`
16. `duracion_horas`
17. `tiempo_al_pico_h`

The contract module owns:

- immutable ordered feature names;
- feature count;
- units and semantic descriptions;
- the exact nullable set: `diam_max_in`, `diam_max_out`,
  `pendiente_max_in`, `pendiente_out`, `dist_outfall_m`, and
  `upstream_capacity_lps`;
- contract ID;
- SHA-256 of a canonical JSON descriptor;
- classification and regression target definitions;
- validation of DataFrame and SQL-result order, type, null policy, and finite
  numeric values.

`factor_mult` remains scenario metadata and is not a model input. Targets and
post-SWMM result columns may not be inputs. A model or dataset with 15, 16, 18,
reordered, missing, duplicated, or renamed features must fail before fitting or
prediction.

SQL `NULL` preserves a physically undefined hydraulic quantity and is not the
same as zero. Required features, including `duracion_horas` and
`tiempo_al_pico_h`, may not be null. Contract-approved nullable values are
imputed by the model pipeline inside each training fold; non-numeric values and
positive/negative infinity are always rejected.

Default values may not manufacture the two shape features during training.
They must be derived from persisted scenario data. A deliberate zero is valid
only when it is physically correct and traceable, not as backward
compatibility for an old row.

## 9. Model Registry And Preprocessing

Supported regression algorithms:

- Ridge;
- Lasso;
- RBF SVR;
- Random Forest regressor;
- XGBoost regressor.

Supported classification algorithms:

- Logistic Regression;
- RBF SVC with probabilities enabled or explicitly calibrated;
- Random Forest classifier;
- XGBoost classifier.

Every candidate receives identical rows, ordered feature columns, groups, and
fold definitions.

Preprocessing rules:

- median imputation is fitted within each training fold;
- StandardScaler is used for linear and SVM families;
- tree models do not require scaling;
- regression volume uses `log1p` inside a fitted target-transform wrapper and
  returns to cubic metres with `expm1`;
- PCA is disabled in the consolidated baseline because it obscures the
  physical feature contract. Reintroducing PCA requires a separate explicit
  design and leakage-safe fold-local implementation;
- all algorithm parameters live under structured `config.yaml` entries;
- no parallel model constants remain in `config.py`.

## 10. Evaluation And Configurable Selection

Evaluation stores OOF predictions so metrics can be recalculated without
retraining when the required observed values, predictions, or probabilities
are already present.

Initial defaults are:

- classification primary metric: `pr_auc`, maximize;
- regression primary metric: `log_nse`, maximize;
- system primary metric: `total_volume_error_pct`, minimize.

These are configuration, not hard-coded selection behavior. A metric registry
defines:

- valid problem types;
- required inputs;
- maximize/minimize direction;
- undefined-value policy;
- deterministic tie-break behavior.

Configuration supports ordered tie-breakers. Changing a reporting/ranking
metric recomputes from stored OOF predictions. Retraining is required only when
the metric controls hyperparameter search, early stopping, threshold tuning,
or another fitting decision.

All evaluations use grouped splits. Rows from one run may not cross training
and validation. LOSO groups reflect the experiment definition (for example
factor/shape scenario groups); GroupKFold groups by run. The exact group IDs
for every fold are persisted.

Reported metrics include, where defined:

- classification: PR-AUC, F1, precision, recall, ROC-AUC, CSI, and confusion
  matrix values;
- regression: MAE, RMSE, NSE, log-NSE, and volume error;
- end-to-end: classification plus node/network total-volume errors;
- stratified results by scenario/factor/shape and extrapolation status.

Interpretability adapts to the model family: coefficients for linear models,
native/SHAP analysis for supported trees, and permutation importance for SVM
or other models without native importance. No analysis path assumes every
model has `feature_importances_`.

## 11. Simulation Write Lifecycle And Recovery

1. Validate database schema, network identity, scenario inputs, units, and
   configuration.
2. Insert or reuse immutable network and scenario source records.
3. Create a `PENDING` run.
4. Mark it `RUNNING` immediately before SWMM execution.
5. Insert aggregate and temporal rows in bounded batches.
6. Build and validate all 17 features from the persisted source data.
7. Verify expected node/timestep counts, key uniqueness, continuity, target
   semantics, and feature contract.
8. Mark the run `COMPLETE` only after all checks succeed.

On controlled failure, child rows are deleted within a transaction and the run
is marked `FAILED` with structured failure details. An unexpected process
termination may leave `RUNNING`; startup recovery marks abandoned runs failed
and removes partial children before those IDs can be reused. Training queries
exclude every status other than `COMPLETE`.

## 12. Inference Contract

Inference chooses a model by explicit `model_id` or by a documented query such
as the best valid model for a target and contract. It never chooses a
filesystem model by conventional filename.

Before deserializing or predicting, it verifies:

- BLOB SHA-256;
- target;
- `tabular_v3_17` ID and descriptor hash;
- exact ordered feature list;
- preprocessing and target transform metadata;
- required library compatibility;
- network/scenario compatibility and extrapolation rules.

Factor-based and arbitrary-hydrograph inference must call the same feature
builder used for persisted training rows. Hydrograph duration and time-to-peak
must be derived using the same semantics in both paths.

## 13. Controlled Legacy Retirement

The old architecture is removed only after replacement behavior is tested.
Expected retirement scope includes:

- package-level `swmm_resilience/main.py` orchestration;
- `swmm_resilience/desktop/`;
- the old database schema/repository implementation after the new schema owns
  every retained responsibility;
- the parallel `ml/train.py`, `ml/preprocessing.py`,
  `ml/predict_from_inp.py`, and `ml/predict_tabular.py` flow after Ridge,
  Lasso, SVR, Logistic Regression, and SVC have moved to the new registry;
- reset and visualization code used only by desktop/legacy SQLite;
- scratch scripts `pruebas_locales.py`, `verificar.py`, and
  `halve_timeseries.py` after a final reference check;
- versioned model binaries and their Git exceptions;
- mandatory CSV dataset production/consumption paths;
- tests that assert removed behavior, replaced by tests for the new boundary.

Shared modules such as `config.py`, `simulation/runner.py`, `utils.py`, and
visualization files must be split or simplified rather than deleted wholesale.

The cleanup must run a repository-wide reference scan after every retirement
phase. No import, CLI flag, config key, test, or current document may point to
a removed interface.

## 14. CNN/LSTM Boundary

CNN/LSTM and temporal surrogate source is valuable future work and is not
deleted. However, the current implementations depend on old SQLite tables,
Parquet artifacts, legacy static features, and filesystem manifests.

During this consolidation:

- operational CLI exposure that suggests the temporal pipeline is ready is
  removed or disabled;
- no automatic fallback to the old schema remains;
- a clear exception/status explains that temporal training requires the later
  SQLite-v17 redesign;
- reusable network classes may remain;
- tests that only verify model tensor behavior may remain;
- integration tests tied to the retired schema are removed or marked as the
  backlog for the later temporal specification, not left passing against an
  unsupported architecture.

The later temporal design must consume `node_timeseries`, `nodes`, `runs`, and
scenario data from SQLite and define its own temporal/static feature contract.
It must not assume that the tabular 17-feature vector is automatically the
correct CNN/LSTM input.

## 15. Documentation Policy

Current documentation after consolidation is limited to:

- `README.md` for architecture and supported commands;
- `QUICKSTART.md` for setup and a small verified workflow;
- a SQLite schema reference;
- a 17-feature contract/data dictionary;
- a reproducible simulation/training/inference procedure;
- a CNN/LSTM status and future-work document;
- `docs/history/retired-architecture.md`, a concise explanation of what was
  removed and why;
- this approved spec and the implementation plan derived from it.

Historical multi-thousand-line plans/specs that describe SQLite legacy,
`dataset_ml.csv`, desktop, 15 features, or filesystem artifacts are removed
rather than archived as operational-looking instructions. Any still-valid
hydraulic or evaluation insight is summarized into current documentation
before its source is removed.

## 16. Verification Strategy

The implementation plan must use TDD for each replacement and retirement
slice. Required verification includes:

### Feature safety

- exact contract ID, count, names, order, units, and descriptor hash;
- rejection of 15, 16, 18, reordered, duplicated, missing, or renamed inputs;
- rejection of target/result leakage;
- proof that both shape features are derived and persisted correctly;
- parity between training and inference feature construction.

### Database safety

- fresh schema and ordered migrations;
- migration checksum mismatch rejection;
- foreign keys and unique constraints;
- run status/recovery behavior;
- batch insert rollback and partial-row cleanup;
- model BLOB round-trip, SHA-256 mismatch, corrupt payload, and incompatible
  manifest rejection;
- backup/checkpoint behavior;
- a synthetic million-timestep load and critical query benchmarks;
- `EXPLAIN QUERY PLAN` assertions or inspected plans for training, scenario,
  temporal-window, and model-selection queries.

### ML behavior

- construction and fit for every approved algorithm;
- algorithm-specific preprocessing;
- fold-local imputation/scaling and no leakage;
- grouped split disjointness;
- OOF persistence and metric recalculation;
- configurable primary metric, direction, invalid values, and tie-breakers;
- family-appropriate interpretability;
- final fit clearly separated from selection metrics;
- inference by model ID from SQLite only.

### Repository safety

- focused tests after each task;
- fast contract/schema suite separated from heavy PyTorch tests;
- complete test suite at phase boundaries;
- `python -m compileall` for supported modules;
- root CLI help and smoke tests;
- reference scans for 15-feature text, removed modules, mandatory CSV paths,
  old artifact filenames, and stale documentation;
- `git status` checks proving user files and the original checkout are
  untouched.

## 17. Implementation Sequence

The implementation plan will decompose work into independently reviewable
phases:

1. Resolve environment pinning and establish fast safety tests.
2. Introduce the canonical 17-feature contract.
3. Build the new SQLite schema, connection policy, repositories, and views.
4. Persist network/scenario/simulation/temporal data into SQLite.
5. Build the unified model and metric registries.
6. Persist OOF predictions, metrics, and model BLOBs.
7. Switch inference, validation, analysis, and visualization to SQLite.
8. Migrate retained linear/SVM capabilities and verify model parity.
9. Establish the explicit CNN/LSTM pending boundary.
10. Retire the legacy pipeline in dependency-ordered slices.
11. Rewrite current documentation and remove obsolete guidance.
12. Run scale, full-suite, reference, and repository cleanliness gates.

Each phase ends with tests and a small reversible commit. Destructive removal
does not begin until the corresponding replacement phase is passing.

## 18. Acceptance Criteria

The consolidation is complete only when:

- every operational tabular fit and prediction uses `tabular_v3_17`;
- a deliberate 15-feature artifact/dataset is rejected by tests;
- simulation inputs, aggregate results, timesteps, provenance, OOF
  predictions, metrics, and selected model BLOBs live in one SQLite database;
- no mandatory training CSV or filesystem model artifact remains;
- every approved linear, SVM, Random Forest, and XGBoost family is available
  through the unified registry;
- primary metrics and tie-breakers are changeable through validated config;
- the legacy desktop, duplicate coordinator, duplicate tabular pipeline, and
  obsolete docs are gone;
- CNN/LSTM is preserved but cannot accidentally run against retired storage;
- clean installation uses a supported Python/dependency combination;
- scale and query-plan checks pass;
- the full surviving test suite passes from a clean checkout;
- the original checkout and untracked user documents remain untouched.
