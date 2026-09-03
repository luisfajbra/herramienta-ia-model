# SQLite V17 Legacy Retirement Implementation Plan

> **SUPERSEDED (2026-09-03):** never executed (zero checkboxes checked).
> The active plan (`docs/FLUJO_ACTUAL.md` §12, minimal path) explicitly
> keeps CSV, `.joblib`, and the legacy desktop GUI — nothing here is being
> retired right now. Do not resume without re-confirming with the project
> owner.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut every retained consumer to the SQLite-v17 interfaces, preserve reusable CNN/LSTM model definitions behind an explicit pending boundary, and delete the superseded 15-feature, CSV, Parquet, desktop, and legacy-SQLite paths.

**Architecture:** Replace consumers before removing their dependencies. SQL-backed reporting operates on explicit run/model IDs, temporal neural-network entry points fail clearly until a later design is implemented, and a repository-wide reference gate proves that no supported command can reach retired code.

**Tech Stack:** Python 3.11, sqlite3, pandas, matplotlib, PyTorch model definitions, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md`

## Global Constraints

- Requires passing phase gates from Plans A, B, and C.
- Delete a legacy file only after every retained responsibility has a tested replacement.
- Never delete or rewrite user files in the original checkout; work only on the isolated cleanup branch.
- Current analysis and visualization read SQLite through explicit queries or receive already queried DataFrames.
- CNN/LSTM tensor model definitions remain, but no supported command may read the retired Parquet/schema/manifest pipeline.
- No current test or document may assert a retired interface.
- `docs/history/retired-architecture.md` is the only place where retired names may remain after the final reference gate.

---

### Task 1: Cut Reporting And Visualizations To SQLite V17

**Files:**
- Create: `swmm_resilience/database/reporting_queries.py`
- Create: `tests/database/test_reporting_queries_v17.py`
- Modify: `swmm_resilience/analysis/factor_comparison.py`
- Modify: `swmm_resilience/analysis/flood_volume.py`
- Modify: `swmm_resilience/analysis/resilience.py`
- Modify: `tests/analysis/test_factor_comparison.py`
- Modify: `tests/test_factor_comparison_cli.py`
- Modify: `tests/test_flood_volume.py`
- Modify: `tests/test_resilience.py`

**Interfaces:**
- Produces:
  - `load_run_results(conn, run_id) -> pd.DataFrame`
  - `load_network_run_summary(conn, network_id) -> pd.DataFrame`
  - `load_model_oof_results(conn, training_run_id, algorithm) -> pd.DataFrame`
- Consumes: Plan A schema and Plan C evaluation IDs

- [ ] **Step 1: Write SQL reporting tests before changing callers**

Seed one network with two complete runs and one failed run. Assert exact output
columns and exclusion behavior:

```python
assert load_run_results(conn, complete_run_id).columns.tolist() == [
    "run_id", "network_id", "scenario_id", "scenario_key", "scenario_kind", "factor_mult", "shape_id",
    "node_id", "coord_x", "coord_y", "inunda", "vol_inundacion_m3",
    "peak_flooding_lps", "flooding_duration_min", "max_depth_m",
    "max_depth_ratio",
]
assert set(load_network_run_summary(conn, network_id)["run_id"]) == {
    complete_run_id, second_complete_run_id
}
assert failed_run_id not in set(load_network_run_summary(conn, network_id)["run_id"])
```

Seed OOF rows and assert the loader requires both `training_run_id` and
`algorithm`; it must not select the latest model implicitly.

- [ ] **Step 2: Implement explicit SQL queries**

`load_run_results()` joins `runs -> scenarios -> node_results -> nodes`,
filters `runs.status='COMPLETE'`, and orders by `nodes.node_id`.
`load_network_run_summary()` returns one row per complete run with node count,
flooded-node count, total volume, factor, duration, and time-to-peak.
`load_model_oof_results()` joins evaluations, OOF rows, nodes, and scenarios,
then orders by fold, run, and node.

All functions accept an existing connection, use bound parameters, and raise
`LookupError` for an unknown or non-complete requested run.

- [ ] **Step 3: Migrate retained analysis functions**

Keep pure DataFrame calculations in `flood_volume.py`, `resilience.py`, and
`analysis/model_comparison.py`. Change CLI/loading boundaries to obtain their
DataFrames from `reporting_queries.py`; remove direct access to legacy tables,
CSV files, and conventional model filenames.

`generate_factor_comparisons()` receives `db_path`, `network_id`, and optional
explicit run IDs. Model-comparison plots receive the DataFrame returned by
`load_model_oof_results()` or by Plan C prediction, rather than loading model
files themselves.

- [ ] **Step 4: Pass queried DataFrames into retained plots**

Analysis commands pass the queried result/OOF DataFrames directly into the
pure plotting functions. Do not route them through
`visualization/loaders.py` or `visualization/runner.py`; both are retired with
the desktop path in Task 3.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/database/test_reporting_queries_v17.py tests/analysis/test_factor_comparison.py tests/test_factor_comparison_cli.py tests/test_flood_volume.py tests/test_resilience.py tests/test_visualization.py -q
rg -n "node_predictions|run_summary|temporal_artifacts|read_csv|classifier\.joblib|regressor\.joblib" swmm_resilience/analysis
git add swmm_resilience/database/reporting_queries.py swmm_resilience/analysis tests/database/test_reporting_queries_v17.py tests/analysis/test_factor_comparison.py tests/test_factor_comparison_cli.py tests/test_flood_volume.py tests/test_resilience.py
git commit -m "refactor: query reports from SQLite v17"
```

Expected: tests pass; the scan has no legacy-table, CSV-read, or file-model
matches in retained reporting code.

### Task 2: Establish The Explicit CNN/LSTM Pending Boundary

**Files:**
- Create: `swmm_resilience/ml/temporal/status.py`
- Modify: `swmm_resilience/ml/temporal/__init__.py`
- Create: `tests/ml/temporal/test_pending_status.py`
- Modify: `swmm_resilience/ml/temporal/models/__init__.py`
- Modify: `swmm_resilience/ml/temporal/models/cnn.py`
- Modify: `swmm_resilience/ml/temporal/models/surrogate_cnn.py`
- Modify: `swmm_resilience/ml/temporal/models/surrogate_lstm.py`
- Modify: `tests/ml/temporal/test_cnn_model.py`
- Keep: `tests/ml/temporal/test_surrogate_cnn.py`
- Keep: `tests/ml/temporal/test_surrogate_lstm.py`
- Delete: `swmm_resilience/ml/temporal/compare_surrogate.py`
- Delete: `swmm_resilience/ml/temporal/dataset.py`
- Delete: `swmm_resilience/ml/temporal/predict.py`
- Delete: `swmm_resilience/ml/temporal/schemas.py`
- Delete: `swmm_resilience/ml/temporal/train_cnn.py`
- Delete: `swmm_resilience/ml/temporal/train_surrogate.py`
- Delete: `tests/ml/temporal/test_compare_surrogate.py`
- Delete: `tests/ml/temporal/test_predict_from_parquet_task_schema.py`
- Delete: `tests/ml/temporal/test_surrogate_dataset.py`
- Delete: `tests/ml/temporal/test_surrogate_predict.py`
- Delete: `tests/ml/temporal/test_surrogate_predict_lstm.py`
- Delete: `tests/ml/temporal/test_temporal_window_summary.py`
- Delete: `tests/ml/temporal/test_train_surrogate.py`
- Delete: `tests/ml/temporal/test_unified_dataset.py`
- Delete: `tests/ml/temporal/test_window_builder.py`

**Interfaces:**
- Produces: `TemporalPipelinePendingError`, `require_sqlite_v17_temporal_support()`
- Consumes: no legacy persistence interface

- [ ] **Step 1: Write the pending-boundary test**

```python
# tests/ml/temporal/test_pending_status.py
import pytest

from swmm_resilience.ml.temporal import (
    TemporalPipelinePendingError,
    require_sqlite_v17_temporal_support,
)


def test_temporal_pipeline_fails_with_actionable_sqlite_v17_message():
    with pytest.raises(TemporalPipelinePendingError, match="node_timeseries"):
        require_sqlite_v17_temporal_support()
```

- [ ] **Step 2: Implement the status boundary**

```python
# swmm_resilience/ml/temporal/status.py
class TemporalPipelinePendingError(NotImplementedError):
    """The neural temporal pipeline has not yet migrated to SQLite v17."""


def require_sqlite_v17_temporal_support() -> None:
    raise TemporalPipelinePendingError(
        "CNN/LSTM training and inference are pending a separate temporal "
        "contract over SQLite v17 node_timeseries, nodes, runs, and scenarios."
    )
```

Export only the status function/error and reusable model classes from
`temporal/__init__.py`. Do not export a training, dataset, or prediction entry
point.

- [ ] **Step 3: Make retained model definitions contract-neutral**

Require explicit `n_temporal_features` and `n_static_features` constructor
arguments in all three model classes. Remove defaults and docstrings that claim
the retired 6/7/21-feature layouts or refer to deleted training functions. Do
not replace them with 17: the later temporal design must define its own input
contract. Export all three classes from `models/__init__.py`.

Rewrite `test_cnn_model.py` to keep only forward-shape, output-range, invalid-
task, and gradient-flow tests. Remove its imports of `TemporalWindowDataset`
and `train_cnn`, and remove filesystem artifact/training integration tests.

- [ ] **Step 4: Remove unsupported temporal integration code and its tests**

Delete the exact files listed above. These files encode Parquet artifacts,
legacy SQLite tables, filesystem manifests, or old feature semantics. Preserve
only tensor-shape/model-forward tests, then prove those model classes still
work:

```powershell
python -m pytest tests/ml/temporal/test_pending_status.py tests/ml/temporal/test_cnn_model.py tests/ml/temporal/test_surrogate_cnn.py tests/ml/temporal/test_surrogate_lstm.py -q
```

- [ ] **Step 5: Prove no operational temporal entry point survives**

```powershell
rg -n "temporal\.(train|predict|dataset)|train_cnn|train_surrogate|predict_from_parquet|temporal_artifacts|parquet_path" main.py swmm_resilience tests -g '*.py'
```

Expected: no matches. The retained status module names the new
`node_timeseries` table but imports no retired module.

- [ ] **Step 6: Commit**

```powershell
git add swmm_resilience/ml/temporal tests/ml/temporal
git commit -m "refactor: isolate CNN LSTM models pending SQLite redesign"
```

### Task 3: Remove Superseded Runtime Paths

**Files:**
- Modify: `main.py`
- Modify: `config.yaml`
- Modify: `swmm_resilience/config.py`
- Modify: `swmm_resilience/ml/__init__.py`
- Modify: `swmm_resilience/database/__init__.py`
- Modify: `swmm_resilience/visualization/__init__.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_config_defaults.py`
- Modify: `tests/test_cli_sqlite.py`
- Delete: `swmm_resilience/main.py`
- Delete: `swmm_resilience/desktop/app.py`
- Delete: `tests/desktop/test_results_tab.py`
- Delete: `swmm_resilience/database/schema.py`
- Delete: `swmm_resilience/database/repository.py`
- Delete: `swmm_resilience/database/queries.py`
- Delete: `tests/database/test_flood_volume_schema.py`
- Delete: `tests/database/test_flood_volume_summary_output.py`
- Delete: `tests/database/test_temporal_artifacts.py`
- Delete: `swmm_resilience/dataset/assembler.py`
- Delete: `swmm_resilience/dataset/validator.py`
- Delete: `swmm_resilience/dataset/__init__.py`
- Delete: `swmm_resilience/analysis/dataset.py`
- Delete: `swmm_resilience/analysis/eda.py`
- Delete: `swmm_resilience/ml/train.py`
- Delete: `swmm_resilience/ml/trainer.py`
- Delete: `swmm_resilience/ml/preprocessing.py`
- Delete: `swmm_resilience/ml/predict.py`
- Delete: `swmm_resilience/ml/predict_from_inp.py`
- Delete: `swmm_resilience/ml/predict_tabular.py`
- Delete: `swmm_resilience/ml/scenario_predict.py`
- Delete: `swmm_resilience/ml/evaluator.py`
- Delete: `swmm_resilience/simulation/batch.py`
- Delete: `tests/analysis/test_dataset_flood_volume_export.py`
- Delete: `tests/ml/test_prediction_volume_output_schema.py`
- Delete: `tests/ml/test_preprocessing_feature_contract.py`
- Delete: `tests/test_dynamic_dataset_validation.py`
- Delete: `tests/test_evaluator.py`
- Delete: `tests/test_ml_trainer_predict.py`
- Delete: `tests/test_scenario_predict.py`
- Delete: `swmm_resilience/reset.py`
- Delete: `swmm_resilience/visualization/loaders.py`
- Delete: `swmm_resilience/visualization/runner.py`
- Delete: `tests/visualization/test_flood_volume_map_contract.py`
- Delete: `halve_timeseries.py`
- Delete: `generate_obsidian_graph.py`
- Delete: `tests/test_obsidian_graph.py`
- Delete: `obsidian/`

**Interfaces:**
- Produces: one supported root CLI over Plans B and C
- Consumes: `run_scenario()`, `train_and_select()`, `predict_factor()`, and SQL reporting interfaces

- [ ] **Step 1: Add a CLI surface test before deletion**

Extend `tests/test_cli_sqlite.py` to assert `python main.py --help` exposes only
the supported flags `--config`, `--simulate`, `--factor`, `--train`,
`--target`, `--predict`, `--classifier-id`, `--regressor-id`,
`--validate-hydrographs`, `--analyze`, `--export-csv`, `--backup`, and
`--backup-path`. `--target` accepts `system`, `inunda`, or
`vol_inundacion_m3`; `--backup` calls Plan A's checkpointed backup operation.
The seven action flags are mutually exclusive, and tests reject action-specific
options without their parent action.
Assert these retired options are absent:

```python
for retired in (
    "--skip-extraction", "--clf-path", "--reg-path", "--parquet",
    "--train-cnn", "--train-surrogate", "--desktop", "--reset",
):
    assert retired not in help_text
```

Also import every public symbol exported by `swmm_resilience.ml` and
`swmm_resilience.database` so stale package exports fail the test.

- [ ] **Step 2: Finish the root CLI cutover**

Remove temporary imports and branches left by Plans B/C. Root commands resolve
all storage through `Config.database`, all training/prediction through the v17
contract, and all model choices through database IDs. Preserve the verified
`--simulate` behavior and its runtime/map output.

Remove the legacy dataset/model/temporal artifact config dataclasses, YAML
keys, and constants, including `DEFAULT_OUTPUT_CSV`, `DEFAULT_DB_FILE`,
filesystem model directories, PCA switches, and parallel model definitions.
Retain only settings consumed by the root SQLite-v17 simulation, unified ML,
validation, analysis, and explicit export commands. Update config tests to
assert rejected unknown legacy keys rather than preserving compatibility.
Remove deleted loader/runner names from `visualization/__init__.py`.

- [ ] **Step 3: Run the pre-deletion replacement gate**

```powershell
python -m pytest tests/test_cli_sqlite.py tests/test_pipeline_sqlite.py tests/ml/test_training_sqlite.py tests/ml/test_prediction_sqlite.py tests/database/test_reporting_queries_v17.py tests/test_simulate_single_factor.py tests/test_hydrograph_batch.py -q
```

Do not continue unless this command has zero failures.

- [ ] **Step 4: Delete the exact superseded files**

Delete the files/directories listed in this task. Remove imports from package
`__init__.py` files. If a repository-wide scan shows a retained responsibility
still depends on one of these files, migrate that caller and rerun Step 3 before
deleting it; do not add a compatibility shim.

- [ ] **Step 5: Run the post-deletion reference and import gates**

```powershell
rg -n "swmm_resilience\.main|swmm_resilience\.desktop|database\.(schema|repository|queries)|dataset\.assembler|simulation\.batch|ml\.(train|trainer|preprocessing|predict_from_inp|predict_tabular|scenario_predict|evaluator)|visualization\.runner|generate_obsidian_graph|halve_timeseries" . -g '*.py' -g '!.git/**'
python -m compileall main.py swmm_resilience
python -m pytest tests/test_cli_sqlite.py tests/test_pipeline_sqlite.py tests/ml tests/database tests/simulation -q
```

Expected: reference scan has no matches, compilation succeeds, and all
surviving focused tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -A
git commit -m "refactor: retire superseded runtime pipelines"
```

### Task 4: Remove Versioned Artifacts And Unused Dependencies

**Files:**
- Modify: `.gitignore`
- Modify: `requirements.txt`
- Delete: `outputs/models/classifier.joblib`
- Delete: `outputs/models/regressor.joblib`
- Delete: `outputs/models/training_inp_hash.txt`
- Delete: `config_7shapes.yaml`
- Delete: `validation_output/`
- Delete: `validation_output_smoke/`
- Modify: `tests/test_dependency_contract.py`

**Interfaces:**
- Produces: reproducible source-only checkout with local SQLite ignored
- Consumes: Plan C in-database model artifacts

- [ ] **Step 1: Extend the dependency/artifact contract test**

Assert:

```python
assert not Path("outputs/models/classifier.joblib").exists()
assert not Path("outputs/models/regressor.joblib").exists()
assert "!outputs/models/classifier.joblib" not in gitignore
assert "!outputs/models/regressor.joblib" not in gitignore
assert "*.sqlite3" in gitignore
assert "*.sqlite3-wal" in gitignore
assert "*.sqlite3-shm" in gitignore
assert "pyarrow" not in requirements
assert "ydata-profiling" not in requirements
assert "Pillow" not in requirements
```

- [ ] **Step 2: Remove tracked generated artifacts**

Delete the three model/hash files and the two checked-in validation-output
directories. Delete `config_7shapes.yaml` after its still-current shape/model
settings have been represented in the canonical `config.yaml` by Plans B/C.

- [ ] **Step 3: Tighten ignore rules**

Remove the three `outputs/models` negations. Add:

```gitignore
*.sqlite3
*.sqlite3-wal
*.sqlite3-shm
.worktrees/
```

Keep the hydrograph-shape CSV allow-list because those CSV files are source
inputs, not operational result storage.

- [ ] **Step 4: Remove dependencies with no retained importer**

After Task 3's reference gate, remove `pyarrow`, `ydata-profiling`, and
`Pillow`. Retain `torch` for CNN/LSTM model definitions and `joblib` for
in-memory SQLite BLOB serialization.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/test_dependency_contract.py tests/ml/temporal/test_cnn_model.py tests/ml/test_artifacts_sqlite.py -q
rg -n "pyarrow|ydata[_-]profiling|PIL|classifier\.joblib|regressor\.joblib|training_inp_hash" . -g '*.py' -g '*.yaml' -g '*.txt' -g '!docs/history/**'
git add -A
git commit -m "chore: remove generated artifacts and obsolete dependencies"
```

Expected: tests pass and the scan has no current-code/config matches.

### Task 5: Replace Historical Documentation With Current References

**Files:**
- Modify: `README.md`
- Modify: `QUICKSTART.md`
- Modify: `comandos.txt`
- Modify: `spec_tecnico_desarrollo.md`
- Modify: `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md`
- Modify: `docs/training_procedure_explained.md`
- Create: `tests/test_current_documentation.py`
- Create: `docs/sqlite_schema.md`
- Create: `docs/feature_contract_v17.md`
- Create: `docs/cnn_lstm_status.md`
- Create: `docs/history/retired-architecture.md`
- Keep: `docs/proximos_estudios.md`
- Keep: `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md`
- Keep: `docs/superpowers/plans/2026-08-21-sqlite-v17-consolidation-roadmap.md`
- Keep: `docs/superpowers/plans/2026-08-21-sqlite-v17-foundation.md`
- Keep: `docs/superpowers/plans/2026-08-21-sqlite-v17-persistence.md`
- Keep: `docs/superpowers/plans/2026-08-21-sqlite-v17-unified-ml.md`
- Keep: `docs/superpowers/plans/2026-08-21-sqlite-v17-legacy-retirement.md`
- Delete: `AUDITORIA_CODIGO_BASURA_2026-06-15.md`
- Delete: `AUDITORIA_SURROGATE_HIDROGRAMAS.md`
- Delete: `AVANCES_RECIENTES_PROYECTO.md`
- Delete: `DOCUMENTACION_COMPLETA_PROYECTO.md`
- Delete: `MODEL_COMPARISON_AND_PREPROCESSING_HISTORY.md`
- Delete: `PLAN_TEMPORAL_LSTM_CNN.md`
- Delete: `REVISION_DETALLADA_DELTA_Y_ESCENARIOS_PARCIALES.md`
- Delete: `XGBOOST_ALGORITHM_OVERVIEW_TRAINING.md`
- Delete: `docs/loso_groupkfold_175_scenarios.md`
- Delete: all pre-2026-08-21 files under `docs/superpowers/plans/`
- Delete: all pre-2026-08-21 files under `docs/superpowers/specs/`

**Interfaces:**
- Produces: one current documentation set and one concise retirement record
- Consumes: final code/config/schema after Tasks 1-4

- [ ] **Step 1: Add documentation link and command tests**

Create `tests/test_current_documentation.py` that extracts repository-relative
Markdown links and `python main.py ...` command examples from every retained
current document listed above. Assert every local target exists and every
command's flags appear in `python main.py --help`.

- [ ] **Step 2: Write the current operational documents**

Document only verified behavior:

- `README.md`: architecture, supported surface, and SQLite as source of truth;
- `QUICKSTART.md`: environment setup, migration, simulation, training,
  prediction, validation, reporting, and optional CSV export;
- `comandos.txt`: copy-pasteable equivalents of the quick-start commands;
- `spec_tecnico_desarrollo.md`: module boundaries and invariants;
- `DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md`: exact ordered 17 features, units,
  nullable set, two targets, and forbidden inputs;
- `docs/training_procedure_explained.md`: grouped OOF, fold-local transforms,
  configurable metrics, selection, BLOB verification, and reproducibility;
- `docs/sqlite_schema.md`: every table/view, key, lifecycle, and query example;
- `docs/feature_contract_v17.md`: contract ID/hash semantics and validation;
- `docs/cnn_lstm_status.md`: retained model classes, disabled integrations, and
  requirements for the later SQLite temporal design.

- [ ] **Step 3: Preserve only useful legacy rationale**

Write `docs/history/retired-architecture.md` as a concise record of the removed
15-feature/CSV/Parquet/desktop paths, why they were unsafe, and which v17
interface replaced each one. Summarize relevant LOSO, temporal, feature, and
Optuna findings there or in `docs/proximos_estudios.md`; do not copy old plans
verbatim.

- [ ] **Step 4: Delete the exact obsolete documents**

Delete the listed root documents and every pre-2026-08-21 plan/spec only after
Step 3 captures still-useful rationale. Retain the approved design, these five
implementation plans, and `docs/proximos_estudios.md`.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/test_current_documentation.py -q
rg -n "dataset_ml\.csv|dataset_final\.csv|classifier\.joblib|regressor\.joblib|temporal_artifacts|parquet_path|15 features|15-feature" README.md QUICKSTART.md comandos.txt spec_tecnico_desarrollo.md DICCIONARIO_DATOS_ENTRENAMIENTO_ML.md docs -g '!docs/history/retired-architecture.md' -g '!docs/superpowers/specs/2026-08-21-*' -g '!docs/superpowers/plans/2026-08-21-*'
git add -A
git commit -m "docs: replace retired architecture references"
```

Expected: documentation test passes; reference scan has no current-document
matches.

### Task 6: Run Final Safety And Acceptance Gates

**Files:**
- Create: `docs/implementation/2026-08-21-sqlite-v17-acceptance.md`
- Verify: complete surviving repository

**Interfaces:**
- Produces: fresh acceptance evidence for branch review
- Consumes: all Plan A-D commits

- [ ] **Step 1: Run the contract and schema safety gate**

```powershell
python -m pytest tests/ml/test_feature_contract_v17.py tests/database/test_migrations_v17.py tests/database/test_training_view_v17.py tests/database/test_simulation_store.py tests/database/test_query_plans_v17.py -q
```

- [ ] **Step 2: Run the model-family and artifact gate**

```powershell
python -m pytest tests/ml/test_model_registry.py tests/ml/test_metric_registry.py tests/ml/test_training_sqlite.py tests/ml/test_artifacts_sqlite.py tests/ml/test_prediction_sqlite.py tests/ml/test_model_explanations.py -q
```

This must exercise Ridge, Lasso, SVR, Logistic Regression, SVC, Random Forest,
and XGBoost through the same ordered v17 contract.

- [ ] **Step 3: Run scale checks**

```powershell
python -m pytest -m scale tests/database/test_query_plans_v17.py -q
```

Record row count, database bytes, insert time, training-view query time, query
plan, and the measured `WITHOUT ROWID` comparison. The shipped schema remains
the ordinary rowid layout documented in Plan A; record whether the result
justifies a separate future migration proposal.

- [ ] **Step 4: Run compile, complete suite, and CLI smoke checks**

```powershell
python -m compileall main.py swmm_resilience
python -m pip check
python -m pytest
python main.py --help
git diff --check
```

Expected: dependency consistency, compilation, tests, and CLI help all succeed,
and the diff check is clean.

- [ ] **Step 5: Run the final retired-reference gate**

```powershell
rg -n "classifier\.joblib|regressor\.joblib|training_inp_hash|dataset_ml\.csv|dataset_final\.csv|temporal_artifacts|parquet_path|swmm_resilience\.main|swmm_resilience\.desktop|FEATURE_COLS\s*=\s*\[" . -g '!docs/history/retired-architecture.md' -g '!docs/superpowers/specs/2026-08-21-*' -g '!docs/superpowers/plans/2026-08-21-*' -g '!.git/**'
```

Expected: no matches.

- [ ] **Step 6: Record evidence and commit**

Write the exact environment, commands, pass count, warning count, runtimes,
scale measurements, database size, reference-scan result, and any accepted
warnings to `docs/implementation/2026-08-21-sqlite-v17-acceptance.md`.

```powershell
git status --short --branch
git log --oneline --decorate --reverse main..HEAD
git add docs/implementation/2026-08-21-sqlite-v17-acceptance.md
git commit -m "docs: record SQLite v17 acceptance evidence"
git status --short --branch
```

Expected: the final status is clean on
`cleanup/sqlite-v17-pipeline-consolidation`. Do not merge or delete the branch;
hand it to the user for review using the finishing-a-development-branch skill.
