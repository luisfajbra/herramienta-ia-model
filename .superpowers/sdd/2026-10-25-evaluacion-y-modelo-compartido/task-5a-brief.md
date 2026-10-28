# Task 5a brief — persist immutable run source artifact links

Plan: `docs/superpowers/plans/2026-10-25-evaluacion-y-modelo-compartido.md`
Spec: `docs/superpowers/specs/2026-10-25-generalizacion-entre-redes-design.md`
Branch: `fix/test-suite-contracts-2026-10-27` (never touch main).

The user approved storing persistent file paths (no copies), registering source artifacts in SQLite plus an experiment manifest, and creating B SQLite `run_id` values before inference while leaving target rows empty until prediction is sealed. This task is only the foundational append-only SQLite registry and its verification API. Do not enable real frozen-transfer scoring or write/update any project datasets/databases.

## Deliverable

- Extend the immutable run-source-artifact table with migration 011. Snapshot run `network_id` and `scenario_id` on each role row, guard run identity/config/status from changing after source registration, and require a linked `source_network_inp` path in addition to effective INP, event source, SWMM RPT and OUT. Keep files in place and persist resolved paths, SHA-256, byte size, canonical source-manifest digest, and registration timestamp.
- Add `swmm_resilience/database/run_source_artifacts.py` with:
  - `register_run_source_artifacts(conn, run_id, artifacts)` which requires exactly one linked file for each required role; resolves paths strictly; hashes current bytes and records size; requires a `COMPLETE` run with immutable hydraulic QA; and requires the RPT hash to match `$.flooding_report.report_sha256`. The linked source-network INP must hash to both `networks.network_sha256` and the bytes in `networks.inp_bytes`. Compute and return a canonical source-manifest digest from run identity plus all role/path/hash/size records; do not accept a caller-asserted digest. Insert atomically and reject malformed/incomplete inputs without partial rows.
  - `verify_run_source_artifacts(conn, run_id)` which reloads every role, checks completeness, run identity snapshots, canonical manifest digest, current files and sizes, source INP/database binding, and RPT/QA link. Return the verified canonical manifest and role records; fail closed on any mismatch.
- Add migration/API tests in `tests/database/test_migration_010_run_source_artifacts.py` and `tests/database/test_run_source_artifacts.py`. Cover five linked files, missing role, modified files, RPT hash mismatch, duplicate registration, SQL immutability, run identity update rejection/tampering, digest recomputation/tampering, and source INP comparison with both stored network hash and INP bytes.
- Update the design spec and implementation plan to record only what this foundation now guarantees; note that source manifests are not yet consumed by `predict_transfer` and `score_transfer` remains unconditionally blocked.

## Existing patterns to follow

- Migrations are contiguous/checksummed. Read `swmm_resilience/database/migrations.py`, `sql/009_run_hydraulic_quality.sql`, `swmm_resilience/database/hydraulic_quality.py`, and `tests/database/test_migration_009_hydraulic_quality.py` before editing.
- Existing run identity is INTEGER in the migration-managed `runs` table. Do not modify historical runs or ignored output SQLite files.
- TDD is mandatory: add one failing test, run it and confirm the intended failure before writing implementation. Re-run relevant database migration tests after green.
- Windows sandbox cannot enumerate pytest temporary directories. Use `--basetemp` and request `require_escalated` for tests if the sandbox returns `WinError 5`; this is a read/test-only action.
- No real SWMM simulation, no model fitting, no target import, no promotion, no score generation.

## Completion

Run the new tests plus existing migration 009 / migration catalog tests and `git diff --check`. Review your diff and commit only Task 5a files with `feat: persist immutable run source artifact links`. Append a short report to `.superpowers/sdd/2026-10-25-evaluacion-y-modelo-compartido/task-5a-implementation-report.md`, including RED test output, GREEN commands/output, review, and commit hash. Do not push; the controller will review first.
