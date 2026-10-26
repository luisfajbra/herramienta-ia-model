# Multi-Network Contract, Identity, and Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing v4_21 path select and preserve one verifiable feature contract and unambiguous network/event identity from SQL through training and inference.

**Architecture:** Keep v17 backward compatible. Resolve feature definitions from the selected contract object, pass that object through preparation/training/loading, and include its ordered columns and descriptor hash in every artifact. Use network-scoped relational keys and explicit run/source selection; add an append-only migration to protect pinned v4_21 source data. Validate candidate pairs before prediction or promotion.

**Tech Stack:** Python, pandas, scikit-learn, joblib, SQLite migrations, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-generalizacion-entre-redes-design.md`, sections “Etapa 1” and “Criterios globales de aceptación”.

## Global Constraints

- Preserve `tabular_v3_17`, legacy tables, and current default CLI behavior.
- The v4_21 route explicitly selects `TABULAR_V4_21`, `FEATURE_COLUMNS_V21`, `training_samples_v4_21`, and `node_features_v4_21` together.
- Network identity is used for keys, selection, manifests and splits; it is not added as a categorical predictor.
- Reading a database must not apply schema migrations or write to the canonical database.
- Do not overwrite active models or canonical datasets. New test databases and candidates live under pytest temporary paths.

---

### Task 1: Resolve candidate metadata from the prepared contract

**Files:** Modify `swmm_resilience/ml/bench/train.py`, `swmm_resilience/ml/bench/schemas.py`, `swmm_resilience/ml/contracts.py`; test `tests/ml/bench/test_train.py`, `tests/ml/test_feature_contract_v21.py`.

**Interfaces:** `PreparedDataset.manifest` supplies `contract_id`, `feature_contract_sha256`, and ordered `feature_columns`; `train_candidate(...)` writes those values, never v17 constants.

- [ ] Add a prepared-dataset fixture with `FEATURE_COLUMNS_V21`, the matching v4_21 contract ID/hash, and valid training labels. Add an assertion that the trained model has 21 inputs and metadata matches all 21 names in order.
- [ ] Run `python -m pytest tests/ml/bench/test_train.py::test_metadata_records_the_prep_id_and_feature_order -q` and the new test. Confirm the v21 test fails because metadata declares v17.
- [ ] In `train_candidate`, derive contract fields exclusively from the validated prepared manifest. Reject unknown contract IDs, a descriptor-hash mismatch, or `X.columns` differing from the declared ordered columns before fitting.
- [ ] In `PreparedDataset.__post_init__`, resolve and validate the registered feature contract rather than treating a missing `feature_columns` field as proof of v17. Keep v17 compatibility for existing complete v17 manifests.
- [ ] Run `python -m pytest tests/ml/bench/test_train.py tests/ml/test_feature_contract_v17.py tests/ml/test_feature_contract_v21.py -q`. Both v17 and v21 must pass; wrong order and wrong hash must fail before fit.
- [ ] Commit only these files with `fix: derive candidate metadata from prepared feature contract`.

### Task 2: Validate candidate model pairs at every load boundary

**Files:** Modify `swmm_resilience/ml/bench/train.py`, `swmm_resilience/ml/bench/promote.py`, `swmm_resilience/ml/scenario_predict.py`, and the existing tabular loader in `swmm_resilience/ml/predict.py`; test `tests/ml/bench/test_verify_promotion.py`, `tests/test_scenario_predict.py`.

**Interfaces:** `load_candidate(directory)` validates both serialized estimators against one metadata document; prediction loaders reject a classifier/regressor pair with different contract IDs, ordered feature lists, or `prep_id`.

- [ ] Add tests with a 21-input estimator and a v17 metadata record, a swapped feature order, and a classifier/regressor pair with different `prep_id`. Each must raise `ModelProvenanceError` (or the module's existing equivalent) before a prediction is returned.
- [ ] Run `python -m pytest tests/ml/bench/test_verify_promotion.py tests/test_scenario_predict.py -q`; confirm the new mismatch cases fail on current direct loads.
- [ ] Validate file existence, SHA-256, feature count, feature order, contract hash, family, and pair identity in `load_candidate` and the promotion verification boundary. Preserve the existing legacy warning behavior only for the documented legacy layout; never silently reinterpret it as v4_21.
- [ ] Make `ScenarioPredictor` use the same verified candidate/promotion loader. Require an explicit documented override for experimental promotion/network mismatch; keep mismatch permission separate from artifact integrity checks.
- [ ] Run the same focused tests plus `tests/ml/bench/test_train.py`. Tampered artifacts and cross-contract pairs must fail; valid v17 fixture and valid v21 candidate load.
- [ ] Commit only these files with `fix: verify model pair contract and provenance on load`.

### Task 3: Make training selection and backfill network-scoped

**Files:** Modify `swmm_resilience/database/training_queries.py`, `swmm_resilience/database/csv_backfill.py`, `main.py`; test `tests/database/test_load_training_frame.py`, `tests/database/test_csv_backfill.py`.

**Interfaces:** Training selection accepts an explicit collection of run IDs or network IDs. The duplicate snapshot key is `(network_id, scenario identity, node_id)`. Backfill keys and source manifests include `network_id`, INP SHA-256 and event-series SHA-256.

- [ ] Build two fixture INPs with the same node IDs and two input rows with the same shape/factor names but distinct network IDs. Add a test that explicit selection returns both networks and one test that a duplicate within the same network/scenario/node is rejected with that full key in the error.
- [ ] Run `python -m pytest tests/database/test_load_training_frame.py tests/database/test_csv_backfill.py -q`; confirm the multi-network fixture currently collides or is incorrectly attributed.
- [ ] Change `_check_single_snapshot` to validate a single network only when the caller explicitly requests a single-network snapshot. For multi-network selection, validate uniqueness by the complete key and include `network_id` in duplicate diagnostics.
- [ ] Carry network ID through `backfill_networks_and_runs` and `backfill_tributary_features` lookup keys. Reject missing or ambiguous node/run mappings instead of silently skipping them. Record configured source path and hashes in each run's provenance.
- [ ] Remove `main.py`'s hard-coded read of `data/training/dataset_final_v4_21.csv`; resolve the source from explicit config/CLI and fail with the resolved path when absent. Do not fallback to another network's CSV.
- [ ] Run the focused tests and `tests/database/test_migration_006_tributary_contract.py` plus `test_migration_007_tributary_training_view.py`. Existing single-network fixtures must remain unchanged.
- [ ] Commit only these files and tests with `fix: scope training data selection by network identity`.

### Task 4: Protect v4_21 rows used by pinned training runs

**Files:** Create `swmm_resilience/database/sql/008_v4_21_provenance_guards.sql`; modify `swmm_resilience/database/migrations.py` only if a separate migration preflight is needed; test new `tests/database/test_migration_008_v4_21_provenance.py`.

**Interfaces:** Migration 008 follows 007 and adds guards to `node_features_v4_21` using the same pin semantics as `node_features` in migration 005.

- [x] Read migrations 005–007 and `tests/database/test_migration_005_pinning.py`. Create tests that apply all migrations, pin a run via `training_run_inputs`, then attempt insert/update/delete of the referenced `node_features_v4_21` row and update the pinned run's continuity. Each mutation must be rejected; the same mutations on an unpinned fixture remain permitted by these guards.
- [x] Run `python -m pytest tests/database/test_migration_008_v4_21_provenance.py -q`; confirm failures on the current catalog because migration 008 and triggers do not exist.
- [x] Add contiguous migration 008 with insert/update/delete guards for the v4_21 feature rows and a guard for quality fields that identify a pinned run. Follow the exact `training_run_inputs` and `run_status` predicates from migration 005; do not edit an already applied migration file.
- [x] Add a migration test that verifies checksum/catalog ordering and that migration 006 schema remains otherwise unchanged.
- [x] Run `python -m pytest tests/database/test_migration_005_pinning.py tests/database/test_migration_006_tributary_contract.py tests/database/test_migration_007_tributary_training_view.py tests/database/test_migration_008_v4_21_provenance.py -q`.
- [x] Commit the migration and tests with `fix: guard pinned v4_21 training provenance`.

### Task 5: Make reads non-mutating and migration writes explicit

**Files:** Modify `swmm_resilience/database/training_queries.py`, `swmm_resilience/ml/bench/preprocess.py`, and the CLI entry points in `main.py`; test `tests/database/test_load_training_frame.py` and `tests/ml/bench/test_stage1_imports.py`.

**Interfaces:** A read/preparation command opens a read-only snapshot and verifies the applied migration catalog; an explicit migration command is the only path that calls `apply_migrations` on the source database.

- [x] Add a test that fingerprints schema and file bytes before/after `load_training_frame`/`prepare_dataset` on a database missing the latest migration; the functions must report the missing migration without changing the database.
- [x] Run `python -m pytest tests/database/test_load_training_frame.py tests/ml/bench/test_stage1_imports.py -q`; confirm the mutation test fails with current implicit migration behavior.
- [x] Separate schema validation from schema mutation. Apply migrations only in the explicit migration/backfill command; have read-only code check the supported schema version and open a SQLite backup with `mode=ro`.
- [x] Document the exact CLI order in `main.py --help` and README: initialize/migrate the experiment DB, import network/run data, then prepare the selected contract. Preserve legacy behavior for legacy entry points that are documented as mutating.
- [x] Run the focused tests and `tests/database/test_migrations_v17.py tests/database/test_migration_preflight_hooks.py -q`.
- [ ] Commit the reader/CLI change and tests with `fix: keep training reads read only`.

### Task 6: Verify one documented v4_21 preparation path end to end

**Files:** Modify `main.py`, `config.yaml`, `swmm_resilience/ml/bench/preprocess.py` only where required by earlier tasks; add `tests/ml/bench/test_multinetwork_preparation.py`; update README usage.

**Interfaces:** The preparation command accepts an explicit database, contract, network/run selection and output directory, and writes a manifest containing source network IDs, INP/event hashes, contract ID/hash, prep ID and exclusions.

- [ ] Add an integration test that migrates a temporary database, imports two networks with identical node/shape/factor labels, prepares v4_21, reloads `PreparedDataset`, and verifies two network IDs and correct feature metadata survive the round trip.
- [ ] Run `python -m pytest tests/ml/bench/test_multinetwork_preparation.py -q`; confirm the test fails on missing CLI selection/manifest behavior.
- [ ] Wire the explicit preparation flags to `prepare_dataset` and persist the resolved source/run selection. Keep `--bench-prepare`'s existing v17 default unchanged unless a separate explicit contract option is present.
- [ ] Execute the CLI against the test fixture only, reload the generated data, then tamper with one contract hash and confirm load fails. The test fixture writes only under `tmp_path`.
- [ ] Run `python -m pytest tests/ml/bench/test_multinetwork_preparation.py tests/ml/bench tests/database -q`.
- [ ] Commit only workflow code, tests and README with `feat: prepare multi-network v4_21 datasets explicitly`.

## Plan 1 completion gate

The synthetic two-network fixture passes through SQL import, v4_21 preparation, candidate training and verified model loading. Identical names across networks remain distinct; a true same-network duplicate, altered source row, contract hash mismatch or mismatched model pair fails before training or prediction. The active model directory and canonical research database remain unchanged.
