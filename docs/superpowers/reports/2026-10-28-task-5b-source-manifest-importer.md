# Task 5b — frozen source manifest importer

## Result

Implemented `swmm_resilience.database.source_manifest_importer.import_frozen_experiment_manifest`. It consumes only `protocol.json` and `batch_progress.json` under an existing experiment root. It checks the byte-level protocol seal, frozen/complete statuses, expected result count, explicit relative paths confined to the root, and SHA-256 for source-network INP, shape CSVs, effective INP, event source, RPT, and OUT. It requires an explicit exact `scenario_key` and joins only by `(networks.network_sha256, scenarios.shape_id, scenarios.factor_mult)`. Missing, duplicate, ambiguous, ineligible, key-mismatched, or hash-mismatched records abort before registration. SQLite registration and output creation use a savepoint; output creation is exclusive and a partial manifest is removed on a write failure.

The created `experiment_source_manifest.json` records the experiment root, protocol and batch hashes, exact run/scenario identities, relative paths, file hashes and sizes, per-run source-manifest hashes, a link-table hash and a canonical manifest hash. Original simulation files are not copied or changed. No canonical or experiment database was opened or modified for this task; only synthetic fixtures were imported.

## RED/GREEN and regression evidence

- RED: `python -m pytest -q tests/database/test_source_manifest_importer.py` under the repository venv failed during collection because the new importer module did not exist yet.
- GREEN focused: `.\venv\Scripts\python.exe -m pytest -q tests/database/test_source_manifest_importer.py --basetemp=outputs/pytest-task5b-atomicity` — **7 passed**.
- Regression: same command with `tests/database/test_source_manifest_importer.py tests/database/test_run_source_artifacts.py tests/database/test_migration_010_run_source_artifacts.py tests/database/test_migration_009_hydraulic_quality.py tests/database/test_migration_005_full_catalog.py` — **32 passed**.
- The Windows pytest invocations required elevated execution because sandboxed pytest could not enumerate its temporary directories (`WinError 5`). Tests use synthetic databases/files only. There are 13 pre-existing matplotlib/pyparsing deprecation warnings.

## Read-only inventory and admission status

Inspected the A requalification and B exploratory `protocol.json` and `batch_progress.json` records. The protocol seal matches the exact protocol bytes in both batches. Both protocols freeze per-shape CSV hashes and provide `shapes_directory`, but no path per shape; the directory is repository-relative, not rooted under each experiment. Both batch result records only give an RPT path/hash. They do not include `scenario_key`, an event-series file/hash, effective-INP path/hash, or OUT path/hash. The protocol INP paths are repository-relative rather than paths rooted within the experiment. A repository search found no other tracked manifest that supplies those missing per-shape paths or per-run artifact links.

Therefore the current historical A/B records cannot pass the importer, and none were imported. Their hashes are preserved as documented in the Task 5 runbook. A new frozen manifest containing explicit links is required before Task 5c. `score_transfer` remains fail-closed.

## Review and commit

Parent review requested atomic cleanup on manifest write errors; the importer removes a partially created manifest and rolls back SQLite links, covered by a synthetic failure-injection test. The first implementation was committed as `965677ab` (`feat: bind frozen simulation manifests to run ids`) on `fix/test-suite-contracts-2026-10-27`.

## Independent review fixes

Review found that the importer checked only `expected_scenarios` count and could accept a different batch scenario at the same count. Protocols must now carry a unique explicit `expected_scenario_keys` list of the exact planned key set; the importer checks equality and duplicates before file/database linking. It does not derive keys from shapes/factors or infer a special key for `base`. A synthetic `base` scenario with an arbitrary explicit key verifies this contract. Review also found that calling the importer inside an outer SQLite transaction could allow a caller rollback after the manifest file was written. The importer now rejects `conn.in_transaction` at entry and documents that it owns the atomic transaction. A test confirms the rejection leaves no artifact rows or output manifest.

- RED: the review-fix tests reproduced both findings: same-count unexpected key was only rejected later by DB key mismatch, while active outer transaction was accepted.
- GREEN: focused importer suite — **11 passed**.
- Regression: importer + source artifact + migrations 009/010/catalog — **36 passed**, 13 existing matplotlib/pyparsing deprecation warnings.
- Review-fix commit: `fix: seal exact frozen scenario key set` (separate from the initial Task 5b feature commit).
