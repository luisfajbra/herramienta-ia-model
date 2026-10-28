# Task 5a implementation report

## RED

Added the API tests first and ran:

```text
venv/Scripts/python.exe -m pytest tests/database/test_migration_010_run_source_artifacts.py tests/database/test_run_source_artifacts.py --basetemp=.pytest-task5a-red -q
```

The expected failure was collection of `test_run_source_artifacts.py` with `ModuleNotFoundError` for the not-yet-implemented `swmm_resilience.database.run_source_artifacts` module.

## GREEN

After adding migration 010 and the registration/verification API, the final focused and regression run was:

```text
venv/Scripts/python.exe -m pytest tests/database/test_migration_010_run_source_artifacts.py tests/database/test_run_source_artifacts.py tests/database/test_migration_009_hydraulic_quality.py tests/database/test_migration_005_full_catalog.py --basetemp=.pytest-task5a-green4 -q
18 passed, 13 warnings
```

The warnings are existing Matplotlib/Pyparsing deprecations. The first run also identified and led to correction of stale catalog expectations (001..009) and fixture transaction setup.

## Review

- Migration 010 adds a strict four-role registry with run FK, field checks, complete-run/QA/report guards, and update/delete immutability triggers.
- Registration resolves and hashes all four current files, validates source-manifest hash and QA, and writes atomically. Verification checks roles, resolved paths, sizes, hashes, and the RPT/QA link.
- Updated the plan/spec to state what the foundation guarantees and to preserve the boundary: manifests are not consumed by `predict_transfer`; `score_transfer` remains unconditionally blocked.
- `git diff --check` passed. No scoring, real data/database, simulation, or model-fitting code was changed.

## Commit

Feature commit: `c3790ab324b415fe9eee5827eb8ee32fb708dbd1` (`feat: persist immutable run source artifact links`).
