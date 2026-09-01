# SQLite V17 Consolidation Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the overlapping CSV/filesystem/legacy-SQLite pipelines with one SQLite-backed pipeline that enforces the exact 17-feature contract.

**Architecture:** Execute four gated plans in order. Each plan produces a passing, independently reviewable state; legacy deletion begins only after its replacement is active and verified.

**Tech Stack:** Python 3.11, SQLite, pandas, scikit-learn, XGBoost, PySWMM, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md`

## Global Constraints

- Operational tabular contract ID is exactly `tabular_v3_17` with the 17 ordered features from the spec.
- SQLite at `database.path` is the single source of truth; default `data/swmm_resilience.sqlite3`.
- No 15-feature artifact, implicit numeric-column discovery, or automatic feature filling is allowed.
- No model binary is versioned in Git; selected models are stored as verified SQLite BLOBs.
- CSV is optional export only; Parquet is not an operational persistence dependency.
- Preserve Ridge, Lasso, SVR, Logistic Regression, SVC, Random Forest, and XGBoost through the unified registry.
- CNN/LSTM remains source-only pending a later SQLite redesign; it must not run against retired storage.
- The original checkout and untracked user documents remain untouched.
- Every destructive removal requires a passing replacement and a repository-wide reference scan.

---

### Task 1: Execute Plan A — Contract And SQLite Foundation

**Files:**
- Read: `docs/superpowers/plans/2026-08-21-sqlite-v17-foundation.md`
- Produce: canonical contract, database migrations/maintenance, and `training_samples_v17`

**Interfaces:**
- Produces: `TABULAR_V3_17`, `connect_database()`, `apply_migrations()`, and `load_training_samples()`
- Consumes: no new interfaces

- [ ] **Step 1: Execute every checkbox in Plan A in order**

Run the focused commands specified by Plan A after every task.

- [ ] **Step 2: Run the Plan A phase gate**

Run:

```powershell
python -m pytest tests/ml/test_feature_contract_v17.py tests/database/test_connection_v17.py tests/database/test_migrations_v17.py tests/database/test_maintenance_v17.py tests/database/test_training_view_v17.py tests/database/test_query_plans_v17.py -q
```

Expected: all tests pass and no production caller has been switched yet.

- [ ] **Step 3: Review and commit the phase gate**

```powershell
git status --short
git log --oneline --decorate -5
```

Expected: after the documentation baseline commits, only Plan A implementation
commits are present on the cleanup branch.

### Task 2: Execute Plan B — Simulation Persistence

**Files:**
- Read: `docs/superpowers/plans/2026-08-21-sqlite-v17-persistence.md`
- Produce: repositories, network/scenario/run/timestep persistence, and atomic recovery

**Interfaces:**
- Consumes: Plan A database and contract interfaces
- Produces: `SimulationStore`, `run_scenario()`, and a SQLite-backed root pipeline

- [ ] **Step 1: Execute every checkbox in Plan B in order**

- [ ] **Step 2: Run the Plan B phase gate**

```powershell
python -m pytest tests/database/test_simulation_store.py tests/test_pipeline_sqlite.py tests/test_simulate_single_factor.py -q
```

Expected: simulations persist complete aggregate and temporal rows; failed runs cannot enter the training view.

- [ ] **Step 3: Confirm no mandatory CSV/Parquet write in the new path**

```powershell
rg -n "to_csv|to_parquet|dataset_final|dataset_ml" swmm_resilience/pipeline.py swmm_resilience/database main.py
```

Expected: no match in the operational persistence path; optional export code is isolated and named as export.

### Task 3: Execute Plan C — Unified ML And SQLite Artifacts

**Files:**
- Read: `docs/superpowers/plans/2026-08-21-sqlite-v17-unified-ml.md`
- Produce: model/metric registries, OOF persistence, BLOB models, and contract-checked inference

**Interfaces:**
- Consumes: Plan A training view and Plan B complete runs
- Produces: `train_and_select()`, `load_verified_model()`, and SQLite-backed predictors

- [ ] **Step 1: Execute every checkbox in Plan C in order**

- [ ] **Step 2: Run the Plan C phase gate**

```powershell
python -m pytest tests/ml/test_model_registry.py tests/ml/test_metric_registry.py tests/ml/test_training_sqlite.py tests/ml/test_artifacts_sqlite.py tests/ml/test_prediction_sqlite.py -q
```

Expected: all approved algorithms share the same 17-column input and group folds; corrupt or stale artifacts are rejected.

- [ ] **Step 3: Run the fast combined safety gate**

```powershell
python -m pytest tests/ml/test_feature_contract_v17.py tests/database/test_training_view_v17.py tests/ml/test_model_registry.py tests/ml/test_artifacts_sqlite.py -q
```

Expected: all tests pass in under the full-suite runtime.

### Task 4: Execute Plan D — Consumer Cutover And Legacy Retirement

**Files:**
- Read: `docs/superpowers/plans/2026-08-21-sqlite-v17-legacy-retirement.md`
- Produce: one CLI, current documentation, explicit temporal boundary, and deleted legacy code/artifacts

**Interfaces:**
- Consumes: all replacement interfaces from Plans A-C
- Produces: final supported repository surface

- [ ] **Step 1: Execute every checkbox in Plan D in order**

- [ ] **Step 2: Run the final reference gate**

```powershell
rg -n "classifier\.joblib|regressor\.joblib|training_inp_hash|dataset_ml\.csv|dataset_final\.csv|temporal_artifacts|parquet_path|swmm_resilience\.main|swmm_resilience\.desktop|FEATURE_COLS\s*=\s*\[" . -g '!docs/history/**' -g '!docs/superpowers/specs/2026-08-21-*' -g '!docs/superpowers/plans/2026-08-21-*' -g '!.git/**'
```

Expected: no operational-code matches; any allowed historical wording exists only in the explicit retirement record.

- [ ] **Step 3: Run the final verification suite**

```powershell
python -m compileall main.py swmm_resilience
python -m pytest
python main.py --help
git diff --check
git status --short --branch
```

Expected: compilation succeeds, the surviving suite has zero failures, CLI help succeeds, diff check is clean, and only intended branch changes exist.

- [ ] **Step 4: Record final acceptance evidence**

Add the exact test count, runtime, scale benchmark, database size, and reference-scan result to the final implementation report. Do not copy baseline numbers; use fresh output.
