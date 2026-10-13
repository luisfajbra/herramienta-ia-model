# Annex 03 — Data and SQL persistence

**Audit:** thesis project `herramienta-ia-model` (ML surrogate model of nodal flooding on SWMM)
**Branch / base commit:** `fix/cierre-auditoria-hidrogramas-20260927` @ `6e8579b` (with uncommitted changes unrelated to this annex, which were not touched)
**Date:** 2026-10-10
**Area:** data, SQLite v17 schema, migrations, provenance, CSV→SQL backfill, workflow lock, reset and quality of the real data.

---

## 1. Scope and method

### 1.1 Code reviewed line by line

| Module | Lines | Role |
|---|---|---|
| `swmm_resilience/database/sql/001…005_*.sql` | 2,107 | Actual v17 schema (tables, CHECK, FK, triggers, views) |
| `swmm_resilience/database/migrations.py` | 226 | Migration engine with checksum |
| `swmm_resilience/database/connection.py` | 36 | Connection policy (FK, WAL, `sha256()`) |
| `swmm_resilience/database/training_queries.py` | 387 | Canonical reader of the training frame |
| `swmm_resilience/database/csv_backfill.py` | 451 | CSV→SQL writing and training evidence chain |
| `swmm_resilience/database/upgrade.py`, `maintenance.py`, `recovery.py`, `workflow_lock.py`, `migration_005_validator.py`, `queries.py` | 371 | Upgrade with backup, recovery and locking |
| `swmm_resilience/database/schema.py`, `repository.py` | 567 | *Legacy* schema (desktop), which **coexists** with v17 |
| `swmm_resilience/reset.py` | 245 | Data deletion |
| `swmm_resilience/dataset/*.py`, `swmm_resilience/analysis/dataset.py` | 179 | Dataset assembly and validation and legacy export |
| `swmm_resilience/extraction/labels.py` | 140 | Label rule (read because of its direct relevance to the data) |
| `main.py` (lines 865-916, 1050-1080) | — | Callers of `--persist-sql` and `load_training_frame` |
| `tests/database/*.py` | 6,338 | Actual coverage (27 files) |

There are no Alembic migrations: the project uses its own `sql/NNN_*.sql` catalog.

Specifications contrasted: `docs/superpowers/specs/2026-08-21-sqlite-v17-pipeline-consolidation-design.md`, `2026-08-22-sqlite-v17-provenance-hardening-design.md` and `2026-09-03-sqlite-v17-training-frame-cutover-design.md`. The 08-21 one was partially replaced by the 09-03 one, which defines a minimal path. Deviations from 08-21 are reported only when 09-03 does not declare them out of scope.

### 1.2 Diagnostics executed (all read-only on the real data)

- The real databases `outputs/training_v17.sqlite3` (55.3 MB) and `outputs/training_v17_19shapes.sqlite3` (27.3 MB) were **always** opened with `file:…?mode=ro&immutable=1`. `immutable=1` avoids touching the `-wal`/`-shm` files. The real `-wal` is 0 bytes, so the main file is complete.
- Before and after the audit, the SHA-256 of both files remained the same (`d6b8ae75afb04dba…` and `bfc3af8c83e4e694…`). The modification dates did not change either (2026-09-15 09:49 and 2026-09-06 20:18). The `-shm` of `training_v17` changed date during the session (16:53 and 17:03), but not because of these scripts: other agents sharing the scratchpad opened the database in parallel.
- The behavior experiments (triggers, reset, idempotency, atomicity) were run on **synthetic databases or copies** created in the scratchpad. The copies were made with the `backup` API from an `immutable` connection.
- `pytest tests/database -p no:cacheprovider --basetemp=<scratchpad>` gave **208 passed, 3 deselected (scale), 28 s**. The full suite and the `-m scale` tests were not run.
- No code, `config.yaml`, `outputs/` or git was modified. `git status` is identical to the initial one.

Python 3.12.10, SQLite 3.49.1 (project venv). The scripts are in section 5.

---

## 2. Findings

Summary by severity: **🔴 0 · 🟠 4 · 🟡 9 · 🔵 8** (21 findings).

| ID | Sev. | File:line | Description | Quantified impact | Reproducible evidence | Recommendation |
|---|---|---|---|---|---|---|
| **D-01** | 🟠 MAJOR | `sql/005_provenance_integrity.sql:991-998` | The provenance *pinning* of `runs` only blocks `UPDATE` of `status` and `network_id`. `scenario_id`, `config_sha256`, `continuity_error_pct`, `node_count` and timestamps of a run used by a `training_run` can be rewritten. `scenario_inflows` is not pinned at all. Spec 08-22 §3.2.1 requires that "the selected `runs` … rows cannot be updated or deleted". | Swapping `scenario_id` between two pinned runs is accepted by the triggers and by `load_training_frame()`. Each (shape, factor, node) is silently relabeled with the volume of another factor, and the `q_pico_nodo` feature, stored in `node_features`, no longer corresponds to the scenario's `factor_mult`. In the synthetic test database 6 of 96 volumes changed without any error. `_check_single_snapshot` only detects duplicates, not permutations. | Script D05 (`d05_swap.py`) and D04-E1. Result: `runs.scenario_id … UPDATE ACEPTADO` (UPDATE ACCEPTED); `scenario_inflows … INSERT/UPDATE/DELETE ACEPTADOS` (INSERT/UPDATE/DELETE ACCEPTED); "filas cuyo vol cambió para la misma clave: 6" (rows whose vol changed for the same key: 6). | Change the trigger to `BEFORE UPDATE ON runs WHEN EXISTS(pinned)` without a column condition (or with `NEW.* IS NOT OLD.*` for all of them). Add *pinning* triggers to `scenario_inflows`. On read, verify `runs.config_sha256 = scenarios.config_sha256` (today it matches in 1000/1000, but nobody checks it). Add a test. |
| **D-02** | 🟠 MAJOR | `csv_backfill.py:147-156`, `sql/001_v17_initial.sql:79-97,132-153` | Runs are inserted directly as `COMPLETE` from the CSV, without evidence of the SWMM simulation. `swmm_version` and `continuity_error_pct` are left NULL. `started_at_utc` and `completed_at_utc` are two calls to `_now()` at backfill time. `node_results.peak_flooding_lps`, `flooding_duration_min`, `max_depth_m` and `max_depth_ratio` are left NULL. `scenario_inflows` (the applied hydrograph) is left empty and so is `links`. Spec 08-21 §7.2/§11 required SWMM version, continuity error and continuity verification before `COMPLETE`. The 09-03 spec accepted the minimal path, but did not explicitly waive continuity. | None of the **1000/1000** runs (nor of the 475/475 of 19shapes) has a SWMM version or a continuity error. In 988/1000 the start and end timestamps are identical, so they are fabricated. 100% of the secondary hydraulic columns is NULL. With the current database it cannot be shown that any of the 1000 simulations satisfies the mass balance. The training pipeline does not read continuity: only the validation harness does (`validation/hydrograph_batch.py:180-318`, 5% threshold). | D02, section "runs por status" (runs by status): `swmm_null=1000 cont_null=1000 same_ts=988`; `peak_null=160000 …`. D01: `scenario_inflows 0`, `links 0`, `node_timeseries 0`. | In the dataset generation, read the continuity error from each simulation's `.rpt`, persist it in `runs.continuity_error_pct` together with `swmm_version`, and reject or flag runs above 5%. Meanwhile, re-run `_read_continuity_error` on the training `.rpt` files, if they are kept (**not verified** that they exist), and document the observed maximum in the thesis. |
| **D-03** | 🟠 MAJOR | `csv_backfill.py:234-235, 248-249, 295, 425-431` | The evidence chain of the persisted model is empty in its reproducibility fields. `hyperparameters_json='{}'`, `preprocessing_json='{}'` and `library_versions_json='{}'`, although the actual model is a `Pipeline(SimpleImputer(median)+XGB(n_estimators, max_depth, lr, subsample…))`. `fit_seconds=0`. `training_query_contract_sha256` is the SHA-256 of the literal `"SELECT * FROM training_samples_v17"`, not of the view definition. The models are trained with the DataFrame **read from the CSV** (`main.py:874,900`), not with the SQL view declared by `query_sql`. Spec 08-22 §3.5 requires equality of the *library-version JSON*, but with `'{}'` on both sides the check passes trivially. | Both databases: 2 `trained_models` with a joblib blob of 0.8-0.96 MB without a recorded xgboost/sklearn version, so the pickle may fail to deserialize or give a different result with another version. Hash of the actual view: `9f6c21c3…`. Recorded hash: `01bc303d…`. A change in the view would not invalidate the provenance. | D03: table `trained_models` and `model_evaluations` (`hyperparameters_json {}`, `fit_seconds 0.0`); line "sha256(definición vista) 9f6c21c3… vs contract 01bc303d…" (sha256(view definition) …). | Serialize the effective hyperparameters (`model.get_params()`), the preprocessing and `{python, numpy, pandas, sklearn, xgboost, joblib}` in the JSONs. Compute the contract over the view's `sqlite_master.sql`. Train `persist_training_run` on `load_training_samples(conn, run_ids)` instead of the CSV. Measure the real timings. |
| **D-04** | 🟠 MAJOR | `extraction/labels.py:19-23,79-86,110-112`; `config.yaml:14` | The flooding volume comes from the `.rpt` in 10⁶ L with 3 decimals, i.e. with **1 m³ resolution**. The label threshold is `flood_threshold_m3: 1.0`, **equal to the resolution**. In practice, `inunda = "SWMM printed ≥ 0.001 ML"`: the threshold filters nothing other than the report's rounding. The regressor's target volume is quantized to integers. | 100% of the 46,242 positive volumes are integers (2,737 unique values). The minimum positive volume is exactly 1.0 m³ and there is none in (0, 1). 7.9% of the positives (3,642) are between 1 and 3 m³, where the relative quantization error reaches 50%. There are 72 1→0 transitions when the factor **increases** (21 nodes, 17 shapes), all with a previous volume of 1-2 m³, plus 79 volume drops. They are rounding and calculation noise at the edge, not physics. | D02 (`frac enteros=1.0000` (integer fraction), `0<vol<thr=0`) and D06 (`vol previo en esas transiciones: {1.0: 68, 2.0: 4}` (previous vol in those transitions)). | State in the thesis that the label is equivalent to "reported volume > 0 with 1 m³ resolution". Evaluate a threshold with a margin above the resolution (e.g. 5-10 m³) or read the volume from the binary `.out`, which is not rounded. Analyze the sensitivity of F1 and of the volume metrics to the threshold. |
| **D-05** | 🟡 MODERATE | `training_queries.py:380-382`; `main.py:888-889` | `load_training_frame()` is a **read** function, but it calls `apply_migrations()` on the real database without `WorkflowLock` and without backup. `--persist-sql` does the same. Spec 08-22 (lines 539-545) requires `upgrade_database_with_backup()` for databases at 001-004 and that "all managed writers must acquire it [lock]". A future migration 006 would be applied without backup on the first `--only-maps` or `--resilience-curve`. | Synthetic database at version 4 → after `load_training_frame()` it ends at **version 5**, with 0 backups created. With the database up to date, the read does not change bytes (D04-E6: identical SHA, no residual sidecars). | D04-E7: "versión de esquema tras 'leer': 5 \| backups creados: []" (schema version after 'reading': 5 \| backups created: []). | On read, check the version and **fail** if there are pending migrations, indicating to run `upgrade_database_with_backup`. Remove `apply_migrations` from the read paths. |
| **D-06** | 🟡 MODERATE | `csv_backfill.py:202, 386, 450`; `main.py:865-916` | `--persist-sql` is neither atomic nor idempotent. (a) The backfill commits (l. 202) before training. (b) `persist_training_run` makes an intermediate commit after the CV (l. 386): if it fails afterwards (final fit, `joblib.dump`, model insertion), a permanent `RUNNING` `training_run` is left, which cannot be deleted because of a trigger and which pins all the runs. (c) Repeating the execution with the same CSV fails with `UNIQUE(scenarios…)`. The only documented remedy is "delete the database file", without backup. `main.py` does not use `try/rollback/close`. | Simulated failure after the intermediate commit: `training_runs=[(1,'RUNNING')]`, 10 COMPLETE evaluations, 192 OOF, 0 models; `DELETE training_runs → 'training runs cannot be deleted'`. Second execution: `IntegrityError UNIQUE constraint failed: scenarios…`. | D04-E4 and D04-E5. | A single `BEGIN IMMEDIATE … COMMIT` for the whole training, or `recover_abandoned_training_run` in an `except`. Make the backfill idempotent (`INSERT … ON CONFLICT DO NOTHING` plus content-equality verification). Back up with `checkpoint_and_backup` before recreating the database. |
| **D-07** | 🟡 MODERATE | `workflow_lock.py` (only used in `upgrade.py:78`) | `WorkflowLock` is only acquired in `upgrade_database_with_backup`. Neither `--persist-sql`, nor `recover_abandoned_training_run` (which delegates it to the caller), nor the 8 reads of `main.py` use it. The write transaction of `persist_training_run` lasts the whole 5-fold CV (≈60 s according to the timestamps of the real database) and blocks other writers, whose `busy_timeout` is 5 s. | Two concurrent `--persist-sql`, or `--persist-sql` + `upgrade`, are not serialized. The second fails with `database is locked` after 5 s and leaves the partial state of D-06. | `grep WorkflowLock swmm_resilience` → only call in `upgrade.py:78`. `training_runs` timestamps in the real database: 14:48:48 → 14:49:49. | Acquire `WorkflowLock` in `--persist-sql`, in the recovery and in `apply_migrations`. |
| **D-08** | 🟡 MODERATE | `reset.py:38-47, 66-78`; `desktop/app.py:835-846` | `reset_db` deletes tables from a *legacy* list whose names **coincide** with v17 tables (`runs`, `node_results`). It disables `PRAGMA foreign_keys`, does not back up and does not check which schema the database has. The GUI allows pointing `db_var` at any file. It only asks for yes/no confirmation (CLI `s/N`, GUI `askyesno`). | On a v17 database **without** *pinning* (a possible state after D-06(a)), it deletes 96 `node_results` and 12 `runs` and leaves **96 FK violations** (orphan `node_features`). On the real databases (100% of runs pinned) the trigger blocks it (`IntegrityError pinned by training provenance`) and the transaction is rolled back without loss. | D04-E3. | Detect the schema (`schema_migrations`) and reject v17 databases. Keep `foreign_keys=ON`. Do `checkpoint_and_backup` before deleting. Remove `--db` for v17 (spec 08-21 §13 already planned to retire the *legacy* reset). |
| **D-09** | 🟡 MODERATE | `database/schema.py:6-122,334-388`; `repository.py:24-28`; `visualization/loaders.py:60-62`; `visualization/runner.py:72-88` | Two incompatible schemas coexist with **homonymous tables** (`runs` with `run_id TEXT` versus `INTEGER`, `node_results` with `node_id/flooded` versus `node_pk/inunda`). The legacy `create_schema()` executed on a v17 database applies DDL in autocommit and then fails halfway. | On a synthetic v17 database: `runs.inflow_multiplier` and `runs.input_source` are added, 6 legacy tables are created (`network_nodes`, `network_links`, `run_inputs`, `link_results`, `run_summary`, `temporal_artifacts`) and it ends with `OperationalError no such column: runs.delta_inflow_lps`. The schema is permanently contaminated, although `load_training_frame` keeps working. | D04-E2. | Add to `create_schema` and `connect_db` the condition "if `schema_migrations` exists, abort". Rename or isolate the legacy schema, or retire it (spec 08-21 §13). |
| **D-10** | 🟡 MODERATE | `csv_backfill.py:73-74`; git `core.autocrlf=true` | `network_sha256 = sha256(bytes of the .inp on disk)`. The `.inp` is in git with LF (`i/lf`) but on disk it has CRLF (`w/crlf`) because of `autocrlf`. The network identity **depends on the platform or checkout configuration**. | Git blob SHA: `1cad5f54fd9b9b0e…`. SHA on disk and in the database: `ea48571b013b9749…`. A clone on Linux/macOS or with `autocrlf=false` would generate another `network_id`. `_check_single_snapshot` would fail when mixing, or the provenance would not match that of the thesis. | `git ls-files --eol` and `sha256sum` (section 5, commands C1). | Add `*.inp -text` (or `binary`) in `.gitattributes` and re-normalize. Alternative: document the canonical hash and its EOL convention. |
| **D-11** | 🟡 MODERATE | `.gitignore:68,82` | The canonical data (`outputs/training_v17*.sqlite3`, `data/training/dataset_final.csv`) are ignored by git and there is no versioned manifest with their hash. The `outputs/models/*.joblib` models are versioned. Rebuilding the database requires re-simulating SWMM. | If the 55 MB database is lost, the only copy of 160,000 labeled rows and of the OOF evidence disappears. The bench manifest `outputs/bench/prepared/af3751551ca5df98` does record `source_snapshot_sha256=a88d3b99…`, which **matches** a backup of the current database. The 19-shape one (`4a48359e7e5b53c6`) does not have that field. | D07. | Version a `DATA_MANIFEST` with the SHA-256 of the database, of the CSV and of the `training_frame_sha256`. Archive the database in external storage (Zenodo or OSF) for the defense. |
| **D-12** | 🟡 MODERATE | `csv_backfill.py:121-128` | `scenarios.config_sha256` only covers `{shape_id, factor_mult, duracion_horas, tiempo_al_pico_h}`. It does not cover the hydrograph ordinates (`scenario_inflows` is empty), nor the SWMM options, nor the `.inp` (this one does go via `network_id`). Two different hydrographs with the same `shape_id`, the same duration and the same time to peak produce the same hash. `scenario_key` uses `factor:.2f`. | The 40 real shapes do not collide today (D02: 0 natural-key duplicates, 0 `scenario_key` collisions), but the hash does not prove which hydrograph was simulated. Combined with D-02, the actual shape applied is not verifiable from the database. | D02 and code reading. | Include in `config_json` the SHA-256 of the shape CSV (normalized ordinates) and of the `[OPTIONS]` used. Populate `scenario_inflows` or, at least, store the hash of the series. |
| **D-13** | 🟡 MODERATE | spec 09-03 §6; `dataset/assembler.py:12-38`; `main.py:1052` | Phase 2 of the *cutover* is not implemented: `assemble_dataset` only writes the CSV, `--export-csv` does not exist and the database is only populated with a manual `--persist-sql`. There are two sources of truth without automatic parity control: the CSV (modified on **2026-10-03**) is newer than the database (**2026-09-15**), and all readers use the database. | Today they are **identical** (verified: 160,000/160,000 rows, 0 differences in the 22 shared columns). If the CSV is regenerated after re-simulating, the commands will keep reading the old database without warning. | D03, section "SQL vs CSV". | Implement Phase 2 or, at the very least, a parity check (CSV hash recorded in the database) that warns on read. |
| **D-14** | 🔵 MINOR | `sql/001_v17_initial.sql` (all tables) | The tables are not `STRICT` (SQLite ≥3.37 allows it). `CHECK(v >= 0)` accepts `'abc'` (TEXT > number) and `+Inf`. NaN is rejected because SQLite converts it to NULL. | The risk is mitigated by the validation on read (`TABULAR_V3_17.validate_frame`, `_validate_targets`). In the real databases there are 0 inf and 0 invalid types. | Command C2 (section 5): `[('abc','text'), (inf,'real')]` accepted. | Migrate `node_features` and `node_results` to `STRICT` or add `CHECK(typeof(x)='real' AND x < 1e308)`. |
| **D-15** | 🔵 MINOR | `training_queries.py:291`; `ml/bench/preprocess.py:47`; `main.py:1057-1068` | The canonical frame order is `ORDER BY run_id, node_id`, where `run_id` is a surrogate key assigned according to the scenario order in the CSV. `main.py` claims that it is equivalent to sorting by `(shape_id, factor_mult, node_id)`. It is **true today**, but only because `base` and the other shapes are assembled in alphabetical order. A shape whose id sorts before `base` would break the equivalence and XGBoost (`subsample`) would train with another row order. | Verified `True` in both databases. Repeated reads are deterministic (D04-E6). | Command C3 (section 5). | Sort by the natural key `(shape_id, factor_mult, node_id)` in the view or the loader, and in the bench's `SORT_KEY`. |
| **D-16** | 🔵 MINOR | `sql/005…:1049-1069`; `sql/001…:215-228` | Indexes are missing for the *pinning* triggers and for the child FKs: `node_features(node_pk)`, `oof_predictions(run_id,node_pk)` and `node_features(node_pk,network_id)`. | `EXPLAIN QUERY PLAN`: `SCAN node_features` (160,000 rows) and `SCAN oof_predictions` (320,000 rows) for each affected `nodes` or `node_results` row. It is performance only. | Command C4 (section 5). | `CREATE INDEX … ON node_features(node_pk, network_id)` and `… ON oof_predictions(run_id, node_pk)`. |
| **D-17** | 🔵 MINOR | `upgrade.py:53-62, 110-116`; `maintenance.py:22-31` | The upgrade's "logical fingerprint" only uses `PRAGMA schema_version` plus the checksums: it does not detect **data** changes between the backup and the migration. It is revalidated outside `BEGIN IMMEDIATE` (spec 08-22, lines 551-553, asks for it to be done inside). The backup is verified with `integrity_check`, but not with `foreign_key_check`. | Race window with writers that do not take the lock (D-07). No incidence observed. | Code reading. | Include counts or a table hash in the fingerprint, revalidate inside the migration transaction and add `foreign_key_check` to the backup. |
| **D-18** | 🔵 MINOR | `extraction/labels.py:110-112`; `dataset/assembler.py:29-31` | Non-numeric volumes of the `.rpt` are converted to 0.0 (`fillna(0.0)`), and nodes without a label after the `merge` receive `inunda=0`, without warning. A `node_id` format mismatch (uppercase, spaces) would silently label nodes as "no flooding". | There is no evidence that it has occurred: 0 label/volume discrepancies. The 48 nodes that never flood (30%) were **not verified** against the `.rpt` files. | Code reading. | Count and warn (or fail) when there are coercions or nodes without a match in the `.rpt`. |
| **D-19** | 🔵 MINOR | `repository.py:31-130`; `analysis/dataset.py:10-68` | *Legacy* path: `save_results` does 4 `to_sql` with independent commits (partial writes), uses `INSERT OR REPLACE` in the topology and disabled FK (`sqlite3.connect` without pragma). The export filters `status='completed'`. | The legacy database `data/training/swmm_resilience.db` does not exist today; the path is inactive and does not affect the thesis. | Code reading and `ls data/training`. | Retire it or isolate it (spec 08-21 §13). |
| **D-20** | 🔵 MINOR | `csv_backfill.py:274-284` | The metric persisted in the database (`f1=0.9658`, `rmse=125.10`) comes from `GroupKFold(5)` by `factor_mult`, with interleaved factors (fold 0 = {1,2,3,4,5}) and **all shapes present in training and validation**. It is not a metric of generalization to unseen shapes (LOSO). | If it were cited in the thesis, it would overestimate the generalization compared with the LOSO protocol. No run overlap between training and validation (0). | D03: "factores de validación por fold" (validation factors per fold). | Label `scope` with the protocol (e.g. `group_kfold5_by_factor`) and do not use this value as the main metric. See the ML annex. |
| **D-21** | 🔵 MINOR | topological features (extraction) | Node `108C` has `n_tuberias_out=1` and `diam_max_out=0.7`, but `pendiente_out=NULL`. It is a feature inconsistency. | 1000 rows (0.6%). It passes the contract because `pendiente_out` is *nullable*. | D06. | Review the type of the outgoing link of `108C` (it probably discharges to an *outfall* or is not a conduit) and document it. |
| **D-22** | 🔵 MINOR | `tests/database/` | Coverage: 208 green tests, but **none** covers: `UPDATE runs.scenario_id` or `config_sha256` on pinned runs (D-01), pinned `scenario_inflows`, failure of `persist_training_run` halfway (D-06), re-execution with the same CSV, `reset_db` on v17 (D-08), legacy `create_schema` on v17 (D-09), migration triggered by a read (D-05) nor EOL dependency (D-10). `test_backfill_reuses_existing_network_on_rerun` uses **disjoint** scenarios, so it does not test real idempotency. | The coverage of triggers 002-005 is broad (more than 4,000 lines of tests), but the real operational paths (CLI) are poorly tested. | `grep "UPDATE runs" tests/database` → no results. | Add regression tests for D-01, D-05, D-06, D-08 and D-09. |

> **Correction note (2026-10-11, spike D-04):** see `../correcciones/03-resolucion-de-la-etiqueta.md`. On **D-04**:
>
> 1. "They are rounding noise" does not hold in general. In the re-simulated sample (one network, 17 simulated scenarios, 15 with overflow; the threshold bound was computed on the 15 with per-node record, 13 with overflow), the `.rpt` volume is a monotonic function of the continuous volume, so every 1→0 transition requires the continuous volume to go down. Of the 72 transitions, 12 were re-simulated, and in all 12 the continuous volume goes down: 7 with drops ≥ 0.3 m³ (among them the 4 with 2 m³), 3 with drops of 0.1 to 0.3 m³ and 2 with drops < 0.1 m³ (borderline cases around 0.5 m³). The other 60 were not verified.
> 2. "The threshold filters nothing other than the rounding": in the sample, the label is deterministic and equivalent to a continuous volume ≥ t, with t in (0.4988, 0.5002] m³. It does filter real overflows smaller than ≈ 0.5 m³: 106 of the 414 nodes with overflow in 6 control scenarios, with a maximum of 0.499 m³. It is a deterministic quantization, not noise.
> 3. "Reading the volume from the binary `.out`" is ruled out with the current 3 min `REPORT_STEP`: the `.out` integral departs from the continuous volume by up to 9.5 m³ per node. The appropriate source is the pyswmm statistic `node.statistics['flooding_volume']`.
> 4. A "threshold with margin (5-10 m³)" does not eliminate the borderline band, it only shifts it. Choosing the threshold is a semantic decision pending for the author and does not fix the resolution.

---

## 3. What is fine (verified)

| # | Aspect | Evidence |
|---|---|---|
| B-01 | Physical and referential integrity of both databases: `PRAGMA integrity_check = ok` and `foreign_key_check = 0` violations. | D01 |
| B-02 | Contiguous migration history 001-005. The stored checksums match the current catalog (`430d13d7…`, `ac73354c…`, `d90c151f…`, `142d235e…`, `bc296815…`). The line-break normalization of `read_text()` makes the checksum immune to CRLF/LF in the `.sql` files. The 005 validator is registered (`validate_before_005`, `37c7a858…`). | D01 and computation with `_migration_catalog` |
| B-03 | The migration engine is atomic: each migration runs in `BEGIN IMMEDIATE` with *rollback*, forbids `BEGIN`/`SAVEPOINT` inside the SQL by means of the *authorizer*, checks the prefix and the checksum, rejects advanced history, runs an FK and integrity *preflight* before 005 and refuses to run with an open transaction. | `migrations.py:86-226`, migration tests green |
| B-04 | Managed connections apply `foreign_keys=ON`, `recursive_triggers=ON` and `journal_mode=WAL`, checking the returned value, plus `busy_timeout`. `sha256()` is registered as deterministic and fails closed with NULL or an invalid type. The network *pinning* trigger fails closed on unmanaged connections (without `sha256()`). | `connection.py`, `005:1117-1129` |
| B-05 | Complete shape × factor × node coverage: 40 × 25 × 160 = **160,000** (and 19 × 25 × 160 = 76,000). 0 missing cells, 0 natural-key duplicates, 1 run per scenario, 0 `scenario_key` collisions, 0 orphans (nodes, features, results, scenarios), 100% of runs pinned by the `training_run`. | D02 |
| B-06 | Label consistency: `inunda == (vol ≥ 1.0 m³)` in 160,000/160,000 rows. 0 negative volumes, 0 inf, 0 NaN in the targets. `tiempo_al_pico_h ≤ duracion_horas` always. | D02 |
| B-07 | Physical coherence of the scale features: `q_pico_nodo = base_inflow × factor` and `q_pico_acum_escalado = q_pico_acum_base × factor` exact (ratio min = max = 1.000000). The 13 static features are invariant per node across the 1000 runs. `nodes` and `node_features` match in elevation, depth and base flow. The NaNs of `diam_max_in`, `pendiente_max_in` and `upstream_capacity_lps` (57,000) correspond exactly to headwater nodes (`n_tuberias_in=0`). | D02 |
| B-08 | **SQL ≡ CSV**: 160,000 rows, outer join 100% `both` and 0 differences in all the shared columns. The 19-shape database is an exact subset of the 40-shape one (0 differences in the 76,000 common keys). | D03 |
| B-09 | The network is self-contained: `network_sha256 == sha256(inp_bytes)` and `inp_bytes` equals the current `.inp` on disk (421,820 bytes). The model blobs match `model_sha256`. The feature contract hash is `56af955c…`, equal to the one published in spec 08-22 §3.5. | D03 |
| B-10 | Complete OOF evidence with no leakage: `included_run_ids_json` equals `training_run_inputs` (1000). There are 160,000 unique OOF per target and 0 runs in training and validation of the same fold. The evaluations are COMPLETE, 5 per task. | D03 |
| B-11 | Reproducible read: two calls to `load_training_frame` return identical frames, in the same order. On an up-to-date database, the read neither modifies bytes nor leaves sidecars. The loader validates cardinality, key symmetry, contract and target domain inside a `SAVEPOINT`, which gives a consistent snapshot in WAL. | D04-E6, `training_queries.py` |
| B-12 | Bench traceability: the manifest `af3751551ca5df98` (160,000 rows) records `source_snapshot_sha256 = a88d3b99…`, which matches a backup taken today of the real database. Two backups of the same source give the same SHA, so the fingerprint is stable. | D07, D04-E6 |
| B-13 | The *pinning* protects `node_results`, `node_features`, `scenarios`, `networks` and `nodes`: `reset_db` on a pinned database is fully rolled back. Append-only evidence tables (OOF, metrics, models, promotions). There is an explicit `RUNNING→FAILED` recovery that does not delete rows. | D04-E3, `005`, `recovery.py` |
| B-14 | `WorkflowLock` uses an OS lock (`msvcrt`/`fcntl`) that is released automatically if the process dies, so there are no orphan locks. The `.workflow.lock` file persists but is harmless. The upgrade does WAL *checkpoint*, backup, `integrity_check` and backup SHA before migrating. | `workflow_lock.py`, `upgrade.py`, tests green |
| B-15 | `export_training_samples_csv` writes atomically (`mkstemp` + `replace`). | `training_queries.py:310-331` |
| B-16 | `pytest tests/database`: **208 passed**, 3 deselected (scale), 28 s. | Local execution |

**Not verified:** the actual continuity error values of the 1000 simulations (they are not in the database; see D-02), the existence of the training `.rpt` files, the `-m scale` tests, the POSIX behavior of `WorkflowLock` and the thread safety of the GUI.

---

## 4. Results of the quality checks on the real data

### 4.1 Inventory (rows per table)

| Table | `training_v17` | `training_v17_19shapes` |
|---|---:|---:|
| networks | 1 | 1 |
| nodes | 160 | 160 |
| links | 0 | 0 |
| scenarios | 1,000 | 475 |
| scenario_inflows | 0 | 0 |
| runs (all COMPLETE) | 1,000 | 475 |
| node_features | 160,000 | 76,000 |
| node_results | 160,000 | 76,000 |
| node_timeseries | 0 | 0 |
| training_runs | 1 | 1 |
| training_run_inputs | 1,000 | 475 |
| model_evaluations | 10 | 10 |
| model_evaluation_runs | 10,000 | 4,750 |
| oof_predictions | 320,000 | 152,000 |
| trained_models | 2 | 2 |
| model_metrics | 2 | 2 |
| candidates/rankings/promotions/selections | 0 | 0 |
| schema_migrations | 5 | 5 |
| Objects: tables / indexes / triggers / views | 33 / 43 / 97 / 4 | 33 / 43 / 97 / 4 |
| integrity_check / foreign_key_check | ok / 0 | ok / 0 |
| WAL header (bytes 18-19) | 2 / 2 | 2 / 2 |

### 4.2 Coverage and keys

| Check | `training_v17` | `19shapes` |
|---|---:|---:|
| Shapes × factors × nodes | 40 × 25 × 160 | 19 × 25 × 160 |
| Rows expected / present | 160,000 / 160,000 | 76,000 / 76,000 |
| Missing cells | 0 | 0 |
| Duplicates (shape, factor, node) | 0 | 0 |
| Scenarios with more than 1 run | 0 | 0 |
| `runs.config_sha256 ≠ scenarios.config_sha256` | 0 / 1,000 | 0 / 475 |
| Factor range (step) | 0.2-5.0 (0.2) | 0.2-5.0 (0.2) |
| Nodes without features / features without result / results without feature | 0 / 0 / 0 | 0 / 0 / 0 |
| Runs not pinned by a training_run | 0 | 0 |

### 4.3 Physical and impossible ranges

| Check | `training_v17` | `19shapes` |
|---|---:|---:|
| `vol_inundacion_m3 < 0` | 0 | 0 |
| `factor_mult ≤ 0` | 0 | 0 |
| `prof_max ≤ 0` | 0 | 0 |
| `q_pico_nodo < 0` | 0 | 0 |
| `tiempo_al_pico_h > duracion_horas` | 0 | 0 |
| inf in features | 0 | 0 |
| `q_pico_acum_escalado < q_pico_nodo` | 0 | 0 |
| `n_in=0` with inlet diameter / `n_in>0` without diameter | 0 / 0 | 0 / 0 |
| NaN `diam_max_in`, `pendiente_max_in`, `upstream_capacity_lps` | 57,000 each (headwater nodes) | 27,075 each |
| NaN `pendiente_out` (node 108C, see D-21) | 1,000 | 475 |
| `vol_inundacion_m3` maximum | 7,418 m³ | 7,126 m³ |
| `duracion_horas` / `tiempo_al_pico_h` | 0.5-15 h / 0.083-11 h | 1-13 h / 0.167-6.5 h |
| `peak_flooding_lps`, `flooding_duration_min`, `max_depth_m` NULL | 160,000 (100%) | 76,000 (100%) |
| `swmm_version` / `continuity_error_pct` NULL | 1,000 / 1,000 | 475 / 475 |
| Runs with `started_at = completed_at` | 988 | 473 |

### 4.4 Class balance

| Metric | `training_v17` | `19shapes` |
|---|---:|---:|
| `inunda=1` / total | 46,242 / 160,000 (**28.90%**) | 22,058 / 76,000 (29.02%) |
| Prevalence per factor (min-max) | 0.0% - 57.67% | 0.0% - 58.03% |
| Factors without any flooded node | 5 (0.2-1.0) = 32,000 rows (20%) | 5 |
| Factors with 100% flooded | 0 | 0 |
| Prevalence per shape (min-max) | 21.85% (`shape_ultra_short_impulse`) - 30.62% (`new_hydrograph`) | 25.15% (`rare_extreme`) - 30.62% |
| Nodes that never flood / that always flood | 48 / 0 (of 160) | 53 / 0 |

### 4.5 Label versus volume (threshold 1.0 m³)

| `inunda` \ volume | vol = 0 | 0 < vol < 1 | vol ≥ 1 |
|---|---:|---:|---:|
| 0 | 113,758 | 0 | 0 |
| 1 | 0 | 0 | 46,242 |

| Quantization and monotonicity metric | `training_v17` | `19shapes` |
|---|---:|---:|
| Discrepancies `inunda ≠ (vol ≥ 1)` | 0 | 0 |
| Fraction of positive volumes that are integers | 100% | 100% |
| Minimum positive volume | 1.0 m³ | 1.0 m³ |
| Positives with 1-3 m³ | 3,642 (7.9%) | n/a |
| Volume quantiles of flooded rows (0 / 25 / 50 / 75 / 99 / 100%) | 1 / 24 / 107 / 343.75 / 3,016 / 7,418 | 1 / 21 / 98 / 313 / 2,569 / 7,126 |
| `inunda` 1→0 transitions when the factor increases | 72 of 153,600 (21 nodes, 17 shapes; previous volume 1-2 m³) | 48 of 72,960 |
| Volume drops when the factor increases (>1 m³) | 79 (7); maximum drop 4 m³ | 50 (3) |

### 4.6 Provenance

| Check | `training_v17` | `19shapes` |
|---|---|---|
| `network_sha256 == sha256(inp_bytes)` | Yes | Yes |
| `inp_bytes ==` current `.inp` on disk | Yes (`ea48571b…`) | Yes |
| SHA of the git blob of the same `.inp` (LF) | `1cad5f54…` (≠, D-10) | same |
| `sha256(model_blob) == model_sha256` | 2/2 | 2/2 |
| `feature_contract_sha256` | `56af955c…` (= spec) | same |
| `training_query_contract_sha256` versus view SHA | `01bc303d…` ≠ `9f6c21c3…` (D-03) | same |
| `hyperparameters_json`, `preprocessing_json`, `library_versions_json` | `{}`, `{}`, `{}` | same |
| Persisted metric (GroupKFold5 by factor) | F1 0.9658; RMSE 125.10 m³ | F1 0.9646; RMSE 88.17 m³ |
| Validation folds (factors) | f0 {1..5}, f1 {0.8..4.8}, f2 {0.6..4.6}, f3 {0.4..4.4}, f4 {0.2..4.2} | same |
| SQL versus `dataset_final.csv` | 0 differences / 160,000 rows | n/a |
| Bench manifest versus the database | `a88d3b99…` = current backup | no snapshot field |

---

## 5. Reproducible scripts

All are read-only on `outputs/` (`?mode=ro&immutable=1`). The D04 and D05 experiments create synthetic databases in a temporary subdirectory next to the script. They are run with `venv\Scripts\python.exe -I <script>` from any directory outside `outputs/`; the scripts use absolute paths to the repository. Copies used in this audit: `%TEMP%\claude\…\scratchpad\auditoria_datos_sql\`.

### D01 — Inventory (`d01_inventario.py`)

```python
"""D01 - Inventario de esquema y conteos (SOLO LECTURA, immutable=1: no toca -wal/-shm)."""
import sqlite3
from pathlib import Path

ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
DBS = [ROOT / "outputs/training_v17.sqlite3", ROOT / "outputs/training_v17_19shapes.sqlite3"]

def ro(p):
    return sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro&immutable=1", uri=True)

for db in DBS:
    c = ro(db)
    print("=" * 80); print(db.name)
    hdr = open(db, "rb").read(100)
    print("header write/read version (2=WAL):", hdr[18], hdr[19])
    for pragma in ("journal_mode", "page_size", "page_count", "freelist_count",
                   "user_version", "schema_version", "foreign_keys", "auto_vacuum"):
        print(f"  PRAGMA {pragma} =", c.execute(f"PRAGMA {pragma}").fetchone()[0])
    print("  integrity_check:", c.execute("PRAGMA integrity_check").fetchall()[:5])
    fk = c.execute("PRAGMA foreign_key_check").fetchall()
    print("  foreign_key_check violaciones:", len(fk), fk[:5])
    for r in c.execute("SELECT version,name,substr(checksum_sha256,1,12),applied_at_utc FROM schema_migrations ORDER BY version"):
        print("   ", r)
    for r in c.execute("SELECT * FROM schema_migration_validators"):
        print("   validator", r[0], r[1], r[2][:12])
    print("   ", c.execute("SELECT type, COUNT(*) FROM sqlite_master GROUP BY type").fetchall())
    for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall():
        print(f"   {t:40s} {c.execute(f'SELECT COUNT(*) FROM \"{t}\"').fetchone()[0]:>10d}")
    c.close()
```

### D02 — Data quality (`d02_calidad.py`)

```python
"""D02 - Chequeos de calidad de datos (SOLO LECTURA, immutable=1)."""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
DBS = [ROOT / "outputs/training_v17.sqlite3", ROOT / "outputs/training_v17_19shapes.sqlite3"]
THRESHOLD = 1.0

def ro(p):
    return sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro&immutable=1", uri=True)

def q(c, sql):
    return pd.read_sql_query(sql, c)

for db in DBS:
    c = ro(db); print("=" * 90); print(db.name)
    print(q(c, "SELECT r.status, s.scenario_kind, COUNT(*) n, MIN(r.node_count) mn, MAX(r.node_count) mx, "
               "SUM(r.swmm_version IS NULL) swmm_null, SUM(r.continuity_error_pct IS NULL) cont_null, "
               "SUM(r.started_at_utc=r.completed_at_utc) same_ts FROM runs r JOIN scenarios s USING(scenario_id) GROUP BY 1,2"))
    print(q(c, "SELECT COUNT(*) esc_multi_run FROM (SELECT scenario_id FROM runs GROUP BY scenario_id HAVING COUNT(*)>1)"))
    print(q(c, "SELECT SUM(r.config_sha256<>s.config_sha256) difieren, COUNT(*) total FROM runs r JOIN scenarios s USING(scenario_id)"))
    print(q(c, "SELECT COUNT(*) dup FROM (SELECT network_id, shape_id, factor_mult FROM scenarios GROUP BY 1,2,3 HAVING COUNT(*)>1)"))
    print(q(c, "SELECT COUNT(*) dup_key FROM (SELECT scenario_key FROM scenarios GROUP BY 1 HAVING COUNT(DISTINCT factor_mult)>1)"))
    f = q(c, "SELECT * FROM training_samples_v17")
    shapes, factors, nodes = (sorted(f[k].unique()) for k in ("shape_id", "factor_mult", "node_id"))
    print("shapes/factores/nodos:", len(shapes), len(factors), len(nodes), "esperado", len(shapes)*len(factors)*len(nodes), "filas", len(f))
    dup = f.groupby(["shape_id", "factor_mult", "node_id"]).size()
    print("duplicados:", int((dup > 1).sum()), "faltantes:",
          len(pd.MultiIndex.from_product([shapes, factors, nodes]).difference(dup.index)))
    na = f.isna().sum(); print(na[na > 0].to_string())
    print(f.drop(columns=["scenario_key", "scenario_kind", "shape_id", "node_id"]).describe().T[["min", "max", "mean"]].to_string())
    checks = {
        "vol<0": (f.vol_inundacion_m3 < 0).sum(), "factor<=0": (f.factor_mult <= 0).sum(),
        "prof_max<=0": (f.prof_max <= 0).sum(), "q_pico_nodo<0": (f.q_pico_nodo < 0).sum(),
        "tpico>duracion": (f.tiempo_al_pico_h > f.duracion_horas).sum(),
        "n_in=0 & diam_in notnull": ((f.n_tuberias_in == 0) & f.diam_max_in.notna()).sum(),
        "n_in>0 & diam_in null": ((f.n_tuberias_in > 0) & f.diam_max_in.isna()).sum(),
        "q_acum_esc < q_pico_nodo": (f.q_pico_acum_escalado < f.q_pico_nodo - 1e-9).sum(),
        "inf": int(np.isinf(f.select_dtypes("number").to_numpy()).sum())}
    for k, v in checks.items(): print(f"  {k:30s} {int(v)}")
    rel = (f.q_pico_acum_escalado / f.q_pico_acum_base) / f.factor_mult
    r2 = (f.q_pico_nodo / (f.base_inflow_lps * f.factor_mult)).replace([np.inf, -np.inf], np.nan)
    print("escalado acum min/max:", rel.min(), rel.max(), "| q_pico_nodo/(qb*f) min/max:", r2.min(), r2.max())
    static = ["elev_fondo", "prof_max", "n_tuberias_in", "n_tuberias_out", "diam_max_in", "diam_max_out",
              "pendiente_max_in", "pendiente_out", "base_inflow_lps", "dist_outfall_m",
              "n_nodos_aguas_arriba", "q_pico_acum_base", "upstream_capacity_lps"]
    print("max nunique estáticos por nodo:", int(f.groupby("node_id")[static].nunique(dropna=False).max().max()))
    print("global inunda:", f.inunda.sum(), len(f), round(f.inunda.mean(), 4))
    by_f = f.groupby("factor_mult").inunda.mean(); by_s = f.groupby("shape_id").inunda.mean()
    by_n = f.groupby("node_id").inunda.mean()
    print("por factor min/max:", by_f.min(), by_f.max(), "factores sin positivos:", int((by_f == 0).sum()))
    print("por forma min/max:", by_s.idxmin(), by_s.min(), by_s.idxmax(), by_s.max())
    print("nodos nunca/siempre:", int((by_n == 0).sum()), int((by_n == 1).sum()))
    exp = (f.vol_inundacion_m3 >= THRESHOLD).astype(int)
    print(pd.crosstab(f.inunda, [f.vol_inundacion_m3 == 0, exp]).to_string(), "discrepancias:", int((f.inunda != exp).sum()))
    pos = f.vol_inundacion_m3[f.vol_inundacion_m3 > 0]
    print("vol>0 n/min/frac enteros/(0,thr):", len(pos), pos.min(), np.isclose(pos, pos.round()).mean(), ((pos > 0) & (pos < THRESHOLD)).sum())
    print("cuantiles:", f.vol_inundacion_m3[f.inunda == 1].quantile([0, .25, .5, .75, .99, 1]).round(3).tolist())
    g = f.sort_values(["shape_id", "node_id", "factor_mult"])
    d = g.groupby(["shape_id", "node_id"]).inunda.diff(); dv = g.groupby(["shape_id", "node_id"]).vol_inundacion_m3.diff()
    print("1->0:", int((d == -1).sum()), "de", int(d.notna().sum()), "| caídas vol:", int((dv < -1e-9).sum()), ">1m3:", int((dv < -1).sum()))
    print(q(c, "SELECT COUNT(*) n, SUM(peak_flooding_lps IS NULL) peak_null, SUM(flooding_duration_min IS NULL) dur_null, "
               "SUM(max_depth_m IS NULL) depth_null FROM node_results"))
    print(q(c, "SELECT (SELECT COUNT(*) FROM nodes WHERE node_pk NOT IN (SELECT node_pk FROM node_features)) nodos_sin_feat, "
               "(SELECT COUNT(*) FROM node_features f LEFT JOIN node_results r USING(run_id,node_pk) WHERE r.run_id IS NULL) feat_sin_res, "
               "(SELECT COUNT(*) FROM node_results r LEFT JOIN node_features f USING(run_id,node_pk) WHERE f.run_id IS NULL) res_sin_feat, "
               "(SELECT COUNT(*) FROM runs WHERE run_id NOT IN (SELECT run_id FROM training_run_inputs)) runs_no_pinned, "
               "(SELECT COUNT(*) FROM scenarios WHERE scenario_id NOT IN (SELECT scenario_id FROM runs)) esc_sin_run"))
    print(q(c, "SELECT SUM(n.invert_elevation_m<>f.elev_fondo) elev, SUM(n.max_depth_m<>f.prof_max) prof, "
               "SUM(n.base_inflow_lps<>f.base_inflow_lps) qb FROM node_features f JOIN nodes n USING(node_pk)"))
    c.close()
```

### D03 — Provenance, hashes and SQL versus CSV (`d03_proveniencia_csv.py`)

```python
"""D03 - Proveniencia, hashes y consistencia SQL vs CSV (SOLO LECTURA, immutable=1)."""
import hashlib, json, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
DB40, DB19 = ROOT / "outputs/training_v17.sqlite3", ROOT / "outputs/training_v17_19shapes.sqlite3"
CSV = ROOT / "data/training/dataset_final.csv"
INP = ROOT / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
KEY = ["shape_id", "factor_mult", "node_id"]
ro = lambda p: sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
sha = lambda b: hashlib.sha256(b).hexdigest()

for db in (DB40, DB19):
    c = ro(db); print("=" * 90, db.name)
    s_db, inp = c.execute("SELECT network_sha256, inp_bytes FROM networks").fetchone()
    print("network_sha==sha(inp_bytes):", s_db == sha(inp), "| inp_bytes==disco:", inp == INP.read_bytes())
    tr = pd.read_sql_query("SELECT * FROM training_runs", c); print(tr.T.to_string())
    print("sha(query_sql)==contract:", sha(tr.query_sql[0].encode()) == tr.training_query_contract_sha256[0])
    view_sql = c.execute("SELECT sql FROM sqlite_master WHERE name='training_samples_v17'").fetchone()[0]
    print("sha(vista):", sha(view_sql.encode())[:16], "contract:", tr.training_query_contract_sha256[0][:16])
    inc = json.loads(tr.included_run_ids_json[0])
    print("included==inputs:", sorted(inc) == [r[0] for r in c.execute("SELECT run_id FROM training_run_inputs ORDER BY run_id")])
    print(pd.read_sql_query("SELECT model_id,target,algorithm,hyperparameters_json,preprocessing_json,target_transform_json,"
                            "library_versions_json,length(model_blob) nbytes FROM trained_models", c).to_string())
    for mid, s, blob in c.execute("SELECT model_id, model_sha256, model_blob FROM trained_models"):
        print("blob ok", mid, sha(blob) == s)
    print(pd.read_sql_query("SELECT model_id, scope, metric_name, value FROM model_metrics", c).to_string())
    print(pd.read_sql_query("SELECT evaluation_id,task,hyperparameters_json,fold_id,status,fit_seconds FROM model_evaluations", c).to_string())
    vf = pd.read_sql_query("SELECT e.fold_id, s.factor_mult FROM model_evaluation_runs m JOIN model_evaluations e USING(evaluation_id) "
                           "JOIN runs r USING(run_id) JOIN scenarios s USING(scenario_id) "
                           "WHERE m.role='validation' AND e.task='classification'", c)
    print(vf.groupby("fold_id").factor_mult.apply(lambda s: sorted(set(np.round(s, 2)))).to_string())
    print("train y val solapados:", c.execute("SELECT COUNT(*) FROM model_evaluation_runs a JOIN model_evaluation_runs b ON "
                                  "a.evaluation_id=b.evaluation_id AND a.run_id=b.run_id AND a.role='train' AND b.role='validation'").fetchone()[0])
    print(pd.read_sql_query("SELECT target, COUNT(*) n, COUNT(DISTINCT run_id||'-'||node_pk) uniq FROM oof_predictions GROUP BY target", c))
    c.close()

c = ro(DB40); sql = pd.read_sql_query("SELECT * FROM training_samples_v17", c); c.close()
csv = pd.read_csv(CSV)
m = sql.assign(factor_mult=sql.factor_mult.round(6)).merge(csv.assign(factor_mult=csv.factor_mult.round(6)),
                                                           on=KEY, how="outer", suffixes=("_sql", "_csv"), indicator=True)
print("merge:", m["_merge"].value_counts().to_dict())
for cc in [k for k in csv.columns if k not in KEY and k + "_sql" in m.columns]:
    a, b = m[cc + "_sql"], m[cc + "_csv"]
    diff = ~((a.isna() & b.isna()) | np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9))
    if diff.sum(): print("dif", cc, int(diff.sum()))
c = ro(DB19); s19 = pd.read_sql_query("SELECT * FROM training_samples_v17", c); c.close()
mm = sql.assign(factor_mult=sql.factor_mult.round(6)).merge(s19.assign(factor_mult=s19.factor_mult.round(6)), on=KEY, suffixes=("_40", "_19"))
print("comunes:", len(mm), {k: int((~np.isclose(mm[k + "_40"], mm[k + "_19"])).sum())
                            for k in ["inunda", "vol_inundacion_m3", "duracion_horas", "tiempo_al_pico_h", "q_pico_nodo"]})
```

### D04 — Behavior experiments on synthetic databases (`d04_experimentos.py`)

```python
"""D04 - Experimentos sobre BD SINTÉTICAS/COPIAS en un directorio temporal junto al script."""
import hashlib, shutil, sqlite3, sys, tempfile
from pathlib import Path

ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "database"))
WORK = Path(tempfile.mkdtemp(prefix="d04_", dir=Path(__file__).resolve().parent))
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database import csv_backfill
from swmm_resilience.database.csv_backfill import backfill_networks_and_runs, persist_training_run
from swmm_resilience.database.schema import create_schema
from swmm_resilience.database.training_queries import load_training_frame
from swmm_resilience.reset import reset_db
from swmm_resilience.config import load_config
from test_csv_backfill import _synthetic_dataset   # 2 formas x 6 factores x 8 nodos = 96 filas

sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

def fresh(name, train=True):
    db, inp = WORK / name, WORK / "net.inp"; inp.write_text("[TITLE]\nsynthetic\n")
    conn = connect_managed_database(db); apply_migrations(conn)
    df = _synthetic_dataset(); info = backfill_networks_and_runs(conn, df, inp, "net")
    if train:
        persist_training_run(conn, df, info["run_id_by_key"], info["node_pk_by_id"], load_config(ROOT / "config.yaml"))
    return conn, db, df, inp, info

def e1():  # pinning incompleto
    conn, *_ = fresh("e1.sqlite3")
    conn.execute("UPDATE runs SET scenario_id=(SELECT scenario_id FROM runs WHERE run_id=2) WHERE run_id=1"); conn.commit()
    print("runs.scenario_id de run fijado: UPDATE ACEPTADO")
    conn.execute("UPDATE runs SET config_sha256=?, continuity_error_pct=99 WHERE run_id=1", ("f" * 64,)); conn.commit()
    conn.execute("INSERT INTO scenario_inflows VALUES (1,1,1,0,0,5.0)")
    conn.execute("UPDATE scenario_inflows SET inflow_lps=999 WHERE scenario_id=1")
    conn.execute("DELETE FROM scenario_inflows WHERE scenario_id=1"); conn.commit()
    print("scenario_inflows de escenario fijado: INSERT/UPDATE/DELETE ACEPTADOS")

def e2():  # create_schema legacy sobre v17
    conn, db, *_ = fresh("e2.sqlite3", train=False); conn.close()
    before = [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(runs)")]
    c = sqlite3.connect(db)
    try: create_schema(c)
    except Exception as e: print("create_schema lanzó:", type(e).__name__, e)
    finally: c.close()
    c = sqlite3.connect(db)
    print("columnas añadidas a runs:", sorted(set(r[1] for r in c.execute("PRAGMA table_info(runs)")) - set(before)))
    print("tablas legacy:", [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN "
          "('network_nodes','network_links','run_inputs','link_results','run_summary','temporal_artifacts')")])

def e3():  # reset_db sobre v17
    conn, db, *_ = fresh("e3u.sqlite3", train=False); conn.close()
    n = reset_db(db); c = sqlite3.connect(db)
    print("sin pinning: borradas", n, "node_features restantes", c.execute("SELECT COUNT(*) FROM node_features").fetchone()[0],
          "violaciones FK", len(c.execute("PRAGMA foreign_key_check").fetchall()))
    conn, db2, *_ = fresh("e3p.sqlite3", train=True); conn.close()
    try: reset_db(db2)
    except Exception as e: print("con pinning:", type(e).__name__, e)

def e4():  # re-ejecución backfill
    conn, db, df, inp, _ = fresh("e4.sqlite3", train=True)
    try: backfill_networks_and_runs(conn, df, inp, "net")
    except Exception as e: print("2a ejecución:", type(e).__name__, e)

def e5():  # fallo tras commit intermedio
    conn, db, df, inp, info = fresh("e5.sqlite3", train=False)
    orig = csv_backfill.joblib.dump
    csv_backfill.joblib.dump = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fallo simulado"))
    try: persist_training_run(conn, df, info["run_id_by_key"], info["node_pk_by_id"], load_config(ROOT / "config.yaml"))
    except RuntimeError as e: print("persist lanzó:", e)
    finally: csv_backfill.joblib.dump = orig
    conn.close(); c = sqlite3.connect(db)
    print(c.execute("SELECT training_run_id, status FROM training_runs").fetchall(),
          c.execute("SELECT COUNT(*) FROM trained_models").fetchone()[0])
    try: c.execute("DELETE FROM training_runs")
    except sqlite3.DatabaseError as e: print("DELETE:", e)

def e6():  # lectura sobre copia del BD real
    copies = []
    for name in ("a", "b"):
        src = sqlite3.connect(f"{(ROOT/'outputs/training_v17_19shapes.sqlite3').as_uri()}?mode=ro&immutable=1", uri=True)
        p = WORK / f"copia_{name}.sqlite3"; dst = sqlite3.connect(p); src.backup(dst); dst.close(); src.close(); copies.append(p)
    print("backups idénticos:", sha_file(copies[0]) == sha_file(copies[1]))
    h0 = sha_file(copies[0]); fr = load_training_frame(copies[0]); fr2 = load_training_frame(copies[0])
    print("sha antes/después:", h0, sha_file(copies[0]), "| lecturas iguales:", fr.equals(fr2),
          "| sidecars:", list(WORK.glob("copia_a.sqlite3-*")))

def e7():  # migración por lectura
    cat = WORK / "cat4"; cat.mkdir()
    for f in sorted((ROOT / "swmm_resilience/database/sql").glob("00[1-4]_*.sql")): shutil.copyfile(f, cat / f.name)
    db = WORK / "e7.sqlite3"; conn = connect_managed_database(db); apply_migrations(conn, migration_dir=cat); conn.close()
    try: load_training_frame(db)
    except Exception as e: print("load_training_frame:", type(e).__name__, e)
    print("versión tras 'leer':", sqlite3.connect(db).execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0])

for f in (e1, e2, e3, e4, e5, e6, e7):
    print("###", f.__name__)
    try: f()
    except Exception as exc: print("EXCEPCIÓN:", type(exc).__name__, str(exc)[:300])
```

### D05 — Silent permutation of `scenario_id` (`d05_swap.py`)

```python
"""D05 - Permutación silenciosa de scenario_id entre dos runs fijados (BD sintética)."""
import sys, tempfile
from pathlib import Path
ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests" / "database"))
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.csv_backfill import backfill_networks_and_runs, persist_training_run
from swmm_resilience.database.training_queries import load_training_frame
from swmm_resilience.config import load_config
from test_csv_backfill import _synthetic_dataset
W = Path(tempfile.mkdtemp(prefix="d05_", dir=Path(__file__).resolve().parent))
inp = W / "n.inp"; inp.write_text("[TITLE]\nx\n")
conn = connect_managed_database(W / "s.sqlite3"); apply_migrations(conn)
df = _synthetic_dataset(); info = backfill_networks_and_runs(conn, df, inp, "n")
persist_training_run(conn, df, info["run_id_by_key"], info["node_pk_by_id"], load_config(ROOT / "config.yaml"))
before = load_training_frame(W / "s.sqlite3")
a, b = 3, 5
sa = conn.execute("SELECT scenario_id FROM runs WHERE run_id=?", (a,)).fetchone()[0]
sb = conn.execute("SELECT scenario_id FROM runs WHERE run_id=?", (b,)).fetchone()[0]
conn.execute("UPDATE runs SET scenario_id=? WHERE run_id=?", (sb, a))
conn.execute("UPDATE runs SET scenario_id=? WHERE run_id=?", (sa, b)); conn.commit(); conn.close()
after = load_training_frame(W / "s.sqlite3")
m = before.merge(after, on=["shape_id", "factor_mult", "node_id"], suffixes=("_antes", "_despues"))
print("filas:", len(after), "| vol cambiado misma clave:", int((m.vol_inundacion_m3_antes != m.vol_inundacion_m3_despues).sum()))
# Resultado observado: filas: 96 | vol cambiado misma clave: 6
```

### D06 — Non-monotonicity and quantization (`d06_detalle.py`)

```python
"""D06 - Detalle de no-monotonía, nodo con pendiente_out NULL y cuantización (SOLO LECTURA immutable=1)."""
import sqlite3
from pathlib import Path
import pandas as pd
ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
c = sqlite3.connect(f"{(ROOT/'outputs/training_v17.sqlite3').as_uri()}?mode=ro&immutable=1", uri=True)
f = pd.read_sql_query("SELECT * FROM training_samples_v17", c)
g = f.sort_values(["shape_id", "node_id", "factor_mult"]).copy()
g["prev_inunda"] = g.groupby(["shape_id", "node_id"]).inunda.shift()
g["prev_vol"] = g.groupby(["shape_id", "node_id"]).vol_inundacion_m3.shift()
bad = g[(g.prev_inunda == 1) & (g.inunda == 0)]
print("1->0:", len(bad), "nodos:", bad.node_id.nunique(), "formas:", bad.shape_id.nunique(), "vol previo:", bad.prev_vol.value_counts().to_dict())
drops = g[g.vol_inundacion_m3 < g.prev_vol - 1e-9]
print("caídas:", len(drops), "máx:", (drops.prev_vol - drops.vol_inundacion_m3).max())
print("pendiente_out NULL:", f.loc[f.pendiente_out.isna(), "node_id"].unique(), f.loc[f.pendiente_out.isna(), "diam_max_out"].unique())
v = f.vol_inundacion_m3[f.vol_inundacion_m3 > 0]
print("positivos 1..3 m3:", int(((v >= 1) & (v <= 3)).sum()), "| valores únicos:", f.vol_inundacion_m3.nunique())
```

### D07 — Bench manifest versus the current database (`d07_snapshot.py`)

```python
"""D07 - ¿El manifest del bench corresponde a la BD actual? (SOLO LECTURA immutable=1; copia temporal)."""
import hashlib, json, sqlite3, tempfile
from pathlib import Path
ROOT = Path(r"C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model")
W = Path(tempfile.mkdtemp(prefix="d07_", dir=Path(__file__).resolve().parent))
for name, man in (("training_v17", "af3751551ca5df98"), ("training_v17_19shapes", "4a48359e7e5b53c6")):
    src = sqlite3.connect(f"{(ROOT/'outputs'/(name+'.sqlite3')).as_uri()}?mode=ro&immutable=1", uri=True)
    p = W / f"{name}.sqlite3"; dst = sqlite3.connect(p); src.backup(dst); dst.commit(); dst.close(); src.close()
    snap = hashlib.sha256(p.read_bytes()).hexdigest()
    flat = (ROOT / "outputs/bench/prepared" / man / "manifest.json").read_text(encoding="utf-8")
    print(name, snap[:16], "en manifest:", snap in flat)
# Resultado: training_v17 a88d3b996b6f0cc8 en manifest: True ; 19shapes 69f2d6c1a35e35e9 en manifest: False
```

### Auxiliary commands

```bash
# C1 - Dependencia del hash de red respecto del EOL (D-10)
git ls-files --eol "data/networks/chico_hydro-qx1/*.inp"           # i/lf  w/crlf
git show "HEAD:data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp" | sha256sum   # 1cad5f54…
sha256sum "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"                     # ea48571b…

# C2 - Afinidad de tipos sin STRICT (D-14)
venv/Scripts/python.exe -c "import sqlite3;c=sqlite3.connect(':memory:');c.execute('create table t(v REAL NOT NULL CHECK(v>=0))');c.execute(\"insert into t values ('abc')\");c.execute('insert into t values (1e999)');print(c.execute('select v,typeof(v) from t').fetchall())"

# C3 - Equivalencia del orden run_id frente a (shape, factor, node) (D-15)
#   SELECT r.run_id, s.shape_id, s.factor_mult FROM runs r JOIN scenarios s USING(scenario_id) ORDER BY r.run_id
#   comparar con el orden de pandas sort_values(['shape_id','factor_mult'])  -> True en ambas bases

# C4 - Planes de consulta de triggers y FK hijas (D-16), sobre una conexión immutable
#   EXPLAIN QUERY PLAN SELECT 1 FROM training_run_inputs JOIN node_features ON node_features.run_id=training_run_inputs.run_id WHERE node_features.node_pk=5
#   -> SCAN node_features ;  EXPLAIN QUERY PLAN SELECT 1 FROM oof_predictions WHERE run_id=1 AND node_pk=5 -> SCAN oof_predictions

# Tests del área (subconjunto, sin caché en el repo)
venv/Scripts/python.exe -m pytest tests/database -q -p no:cacheprovider --basetemp=<scratchpad>/pytest-db   # 208 passed, 3 deselected
```
