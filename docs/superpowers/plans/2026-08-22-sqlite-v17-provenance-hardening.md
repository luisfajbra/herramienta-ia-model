# SQLite V17 Provenance Hardening Implementation Plan

> **PARTIALLY SUPERSEDED (2026-09-03):** migration 005 and its guards did
> ship and are live (`csv_backfill.py` follows this lifecycle exactly).
> Whatever here goes beyond migration 005 (full candidate/ranking/promotion
> writers) was not built — see `docs/FLUJO_ACTUAL.md` §12 for the active
> plan before resuming any unchecked step below.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the seven confirmed integrity gaps in the SQLite V17 foundation (migrations 001-004) by adding migration 005, so that every model, metric, promotion, or prediction stays traceable to the exact immutable training inputs and OOF evidence that produced it.

**Architecture:** Add three small pieces of infrastructure the migration system doesn't have yet (a per-migration Python preflight hook, a managed connection that registers a `sha256()` SQL function, and a cross-process advisory workflow lock), then add append-only migration `005_provenance_integrity.sql` that: replaces the rowid-unsafe identity trigger from 004 with one that observes every update; adds normalized membership tables so training/evaluation provenance is verified by joins instead of parsed JSON; adds append-only OOF/candidate/ranking/promotion evidence chain tables; and deterministically invalidates every pre-005 training run and promotion (their evidence was mutable under the old schema and can't be proven after the fact). Finally, add the operational wrappers migration 005 assumes exist: `upgrade_database_with_backup()` and an explicit interrupted-run recovery operation.

**Tech Stack:** Python 3.11, sqlite3 (stdlib), pytest. No new third-party dependencies.

## Global Constraints

- Migrations 001 through 004 remain byte-identical — 005 is strictly additive/append-only at the SQL level (existing tables get new triggers and columns are never removed).
- One training run tests at most one hyperparameter configuration per `(task, algorithm, fold)` — `model_evaluations` keeps its existing `UNIQUE(training_run_id, task, algorithm, fold_id)` unchanged. Comparing multiple hyperparameter variants of the same algorithm requires separate training runs, each producing its own candidate. (Confirmed consistent with `docs/superpowers/specs/2026-08-04-optuna-hyperparam-search-design.md`, whose `apply_if_better` only ever writes one final refit per training run.)
- No compatibility path may silently weaken a guard. A raw (non-managed) connection missing the `sha256()` function must fail closed at the trigger that needs it, not skip the check.
- `node_timeseries` is out of scope — not pinned by this migration.
- Do not begin model training, prediction, CNN-LSTM migration, or legacy pipeline deletion in this work (matches spec section 8, Non-goals).
- All work happens on branch `cleanup/sqlite-v17-pipeline-consolidation`, worktree `C:\Users\Luis\AppData\Local\Temp\herramienta-ia-model-sqlite-v17-cleanup`.

Spec: `docs/superpowers/specs/2026-08-22-sqlite-v17-provenance-hardening-design.md`.

---

## Task 1: Per-migration Python preflight hook

**Files:**
- Modify: `swmm_resilience/database/migrations.py`
- Test: `tests/database/test_migration_preflight_hooks.py`

**Interfaces:**
- `class MigrationPreflightError(RuntimeError)` — raised by a hook to abort before any DDL runs.
- `apply_migrations(conn, migration_dir=None, preflight_hooks: dict[int, Callable[[sqlite3.Connection], None]] | None = None)` — for each migration version present in `preflight_hooks`, the hook runs immediately after `BEGIN IMMEDIATE` for that version and before its first SQL statement. A hook that raises aborts the transaction; nothing from that migration (or later ones) is applied.

This is generic infrastructure — no version 005 wiring in this task. Task 10 wires the real 005 validator through this mechanism.

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_preflight_hooks.py
from pathlib import Path
import shutil
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import (
    MigrationPreflightError,
    apply_migrations,
)

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def _catalog(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    shutil.copyfile(SQL_DIR / "001_v17_initial.sql", catalog / "001_v17_initial.sql")
    (catalog / "002_noop.sql").write_text("SELECT 1;", encoding="utf-8")
    return catalog


def test_preflight_hook_runs_before_ddl_and_can_abort(tmp_path):
    catalog = _catalog(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    calls = []

    def failing_hook(hook_conn):
        calls.append(
            hook_conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        )
        raise MigrationPreflightError("refuse to apply 002")

    with pytest.raises(MigrationPreflightError):
        apply_migrations(
            conn,
            migration_dir=catalog,
            preflight_hooks={2: failing_hook},
        )

    assert calls == [1]  # ran after 001 committed, before 002's DDL
    applied = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert [row[0] for row in applied] == [1]
    conn.close()


def test_preflight_hook_passing_allows_migration_to_apply(tmp_path):
    catalog = _catalog(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog, preflight_hooks={2: lambda c: None})
    applied = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert [row[0] for row in applied] == [1, 2]
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_migration_preflight_hooks.py -v`
Expected: FAIL — `ImportError: cannot import name 'MigrationPreflightError'`

- [ ] **Step 3: Implement the hook mechanism**

In `swmm_resilience/database/migrations.py`, add near the other exception classes:

```python
class MigrationPreflightError(RuntimeError):
    pass
```

Change the `apply_migrations` signature and loop body:

```python
def apply_migrations(
    conn: sqlite3.Connection,
    migration_dir: Path | None = None,
    preflight_hooks: dict[int, "Callable[[sqlite3.Connection], None]"] | None = None,
) -> None:
    if conn.in_transaction:
        raise RuntimeError(
            "Cannot apply migrations while the connection has an active transaction"
        )

    hooks = preflight_hooks or {}
    catalog = _migration_catalog(migration_dir)
    applied = _applied_history(conn)
    _validate_applied_history(applied, catalog)

    applied_any = False
    for version, name, sql, checksum in catalog[len(applied):]:
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute("BEGIN IMMEDIATE")
            hook = hooks.get(version)
            if hook is not None:
                hook(conn)
            _execute_migration_sql(conn, sql)
            conn.execute(
                """
                INSERT INTO schema_migrations (
                    version, name, checksum_sha256, applied_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (version, name, checksum, stamp),
            )
            conn.commit()
            applied_any = True
        except Exception:
            conn.rollback()
            raise

    if applied_any:
        conn.execute("PRAGMA optimize")
```

Add `from typing import Callable` to the imports at the top of the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_migration_preflight_hooks.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run full migrations suite to check for regressions**

Run: `python -m pytest tests/database/test_migrations_v17.py -v`
Expected: PASS (all existing tests, unaffected by the new optional parameter)

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/migrations.py tests/database/test_migration_preflight_hooks.py
git commit -m "feat: add optional per-migration Python preflight hook"
```

---

## Task 2: Cross-process advisory workflow lock

**Files:**
- Create: `swmm_resilience/database/workflow_lock.py`
- Test: `tests/database/test_workflow_lock.py`

**Interfaces:**
- `class WorkflowLockError(RuntimeError)`
- `class WorkflowLock` — context manager. `WorkflowLock(database_path).acquire()` opens `<database_path>.workflow.lock`, takes an OS advisory exclusive lock (non-blocking), and raises `WorkflowLockError` if already held. Released automatically on `__exit__`, process exit, or crash (OS releases the descriptor's lock on process death).

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_workflow_lock.py
import multiprocessing
import time

import pytest

from swmm_resilience.database.workflow_lock import WorkflowLock, WorkflowLockError


def test_second_acquire_in_same_process_fails(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    with WorkflowLock(db_path):
        with pytest.raises(WorkflowLockError):
            with WorkflowLock(db_path):
                pass


def test_lock_is_released_on_exit(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    with WorkflowLock(db_path):
        pass
    with WorkflowLock(db_path):
        pass  # must not raise; prior lock was released


def _hold_lock_then_signal(db_path, ready_event, release_event):
    with WorkflowLock(db_path):
        ready_event.set()
        release_event.wait(timeout=5)


def test_second_process_cannot_acquire_held_lock(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    ready = multiprocessing.Event()
    release = multiprocessing.Event()
    holder = multiprocessing.Process(
        target=_hold_lock_then_signal, args=(db_path, ready, release)
    )
    holder.start()
    try:
        assert ready.wait(timeout=5)
        with pytest.raises(WorkflowLockError):
            with WorkflowLock(db_path):
                pass
    finally:
        release.set()
        holder.join(timeout=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_workflow_lock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swmm_resilience.database.workflow_lock'`

- [ ] **Step 3: Implement the lock**

```python
# swmm_resilience/database/workflow_lock.py
from __future__ import annotations

import os
from pathlib import Path


class WorkflowLockError(RuntimeError):
    pass


class WorkflowLock:
    """Advisory, cross-process exclusive lock over one database file.

    Used to serialize training/migration/recovery operations against a
    single SQLite database. Not a substitute for SQLite's own locking —
    this guards multi-statement Python-level workflows.
    """

    def __init__(self, database_path: str | Path):
        self._lock_path = Path(f"{database_path}.workflow.lock")
        self._fd: int | None = None

    def acquire(self) -> "WorkflowLock":
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR)
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise WorkflowLockError(
                        f"Workflow lock already held: {self._lock_path}"
                    ) from exc
            else:
                import fcntl

                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise WorkflowLockError(
                        f"Workflow lock already held: {self._lock_path}"
                    ) from exc
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "WorkflowLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_workflow_lock.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/database/workflow_lock.py tests/database/test_workflow_lock.py
git commit -m "feat: add cross-process advisory workflow lock"
```

---

## Task 3: Managed connection with registered SHA-256 function

**Files:**
- Modify: `swmm_resilience/database/connection.py`
- Test: `tests/database/test_connection_v17.py`

**Interfaces:**
- `connect_managed_database(path) -> sqlite3.Connection` — calls `connect_database(path)`, then registers a deterministic SQL function `sha256(blob_or_text)` (`conn.create_function("sha256", 1, ..., deterministic=True)`) that hashes UTF-8-encoded text or raw bytes and returns the lowercase hex digest. Triggers that need to verify a hash (e.g. `network_sha256 == sha256(inp_bytes)`, per spec 3.2.1) call this SQL function; a raw connection from `connect_database` simply doesn't have it, so any trigger referencing `sha256(...)` raises `sqlite3.OperationalError: no such function: sha256` on that connection — i.e. it fails closed, not open.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/database/test_connection_v17.py
import hashlib

import pytest

from swmm_resilience.database.connection import connect_managed_database


def test_managed_connection_registers_sha256_function(tmp_path):
    conn = connect_managed_database(tmp_path / "db.sqlite3")
    try:
        digest = conn.execute("SELECT sha256(?)", (b"hello",)).fetchone()[0]
        assert digest == hashlib.sha256(b"hello").hexdigest()
        text_digest = conn.execute("SELECT sha256(?)", ("hello",)).fetchone()[0]
        assert text_digest == hashlib.sha256(b"hello").hexdigest()
    finally:
        conn.close()


def test_raw_connection_has_no_sha256_function(tmp_path):
    from swmm_resilience.database.connection import connect_database

    conn = connect_database(tmp_path / "db.sqlite3")
    try:
        with pytest.raises(Exception):
            conn.execute("SELECT sha256(?)", (b"hello",))
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_connection_v17.py -v`
Expected: FAIL — `ImportError: cannot import name 'connect_managed_database'`

- [ ] **Step 3: Implement**

Append to `swmm_resilience/database/connection.py`:

```python
import hashlib


def _sha256_sql_function(value) -> str:
    if value is None:
        raise ValueError("sha256() requires a non-NULL argument")
    payload = value if isinstance(value, (bytes, bytearray)) else str(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def connect_managed_database(path: str | Path) -> sqlite3.Connection:
    conn = connect_database(path)
    conn.create_function("sha256", 1, _sha256_sql_function, deterministic=True)
    return conn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_connection_v17.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/database/connection.py tests/database/test_connection_v17.py
git commit -m "feat: add managed connection with sha256 SQL function"
```

---

## Task 4: Migration 005 Part A — validator ledger, rowid-safe identity, invalidation infra

**Files:**
- Create: `swmm_resilience/database/sql/005_provenance_integrity.sql`
- Test: `tests/database/test_migration_005_identity.py`

Start the new migration file. Every later task in this plan appends more SQL to the same file (005 is one migration; the split across tasks is only for reviewable, testable increments — the file is not runnable/valid SQLite until Task 9 finishes it, so tests in Tasks 4-8 apply it as a **partial standalone file copied into an isolated test catalog**, the same pattern `test_migrations_v17.py` already uses for 002/003 subsets).

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_005_identity.py
from pathlib import Path
import shutil

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def _catalog_through_005(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql",
        "002_model_integrity.sql",
        "003_model_integrity_guards.sql",
        "004_training_run_identity.sql",
        "005_provenance_integrity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    return catalog


def _insert_training_run(conn, training_run_id=1, target="system", status="PENDING"):
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (?, ?, 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', ?)
        """,
        (training_run_id, target, "a" * 64, status),
    )


def test_rowid_alias_update_is_rejected(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _insert_training_run(conn)
    conn.commit()

    for alias in ("rowid", "_rowid_", "oid"):
        with pytest.raises(Exception):
            conn.execute(f"UPDATE training_runs SET {alias} = 999 WHERE training_run_id = 1")
        conn.rollback()


def test_migration_005_invalidates_every_pre005_training_run(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql",
        "002_model_integrity.sql",
        "003_model_integrity_guards.sql",
        "004_training_run_identity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _insert_training_run(conn, training_run_id=1, status="COMPLETE")
    conn.commit()

    shutil.copyfile(
        SQL_DIR / "005_provenance_integrity.sql",
        catalog / "005_provenance_integrity.sql",
    )
    apply_migrations(conn, migration_dir=catalog)

    invalidated = conn.execute(
        "SELECT reason FROM training_run_provenance_invalidations WHERE training_run_id = 1"
    ).fetchone()
    assert invalidated is not None
    assert invalidated[0] == "pre005_mutable_provenance"
    valid_rows = conn.execute(
        "SELECT 1 FROM valid_training_runs WHERE training_run_id = 1"
    ).fetchall()
    assert valid_rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_migration_005_identity.py -v`
Expected: FAIL — `swmm_resilience/database/sql/005_provenance_integrity.sql` does not exist

- [ ] **Step 3: Create the migration file with Part A**

```sql
-- swmm_resilience/database/sql/005_provenance_integrity.sql

CREATE TABLE schema_migration_validators (
    version INTEGER PRIMARY KEY
        REFERENCES schema_migrations(version) ON DELETE RESTRICT,
    validator_name TEXT UNIQUE NOT NULL,
    validator_sha256 TEXT NOT NULL CHECK(length(validator_sha256)=64)
);

CREATE TRIGGER schema_migration_validators_identity_conflict
BEFORE INSERT ON schema_migration_validators
WHEN EXISTS (
    SELECT 1 FROM schema_migration_validators WHERE version=NEW.version
)
BEGIN
    SELECT RAISE(ABORT, 'schema migration validator identity is immutable');
END;

CREATE TRIGGER schema_migration_validators_immutable_update
BEFORE UPDATE ON schema_migration_validators
BEGIN
    SELECT RAISE(ABORT, 'schema migration validators are immutable');
END;

CREATE TRIGGER schema_migration_validators_immutable_delete
BEFORE DELETE ON schema_migration_validators
BEGIN
    SELECT RAISE(ABORT, 'schema migration validators are immutable');
END;

CREATE TABLE training_run_provenance_invalidations (
    training_run_id INTEGER PRIMARY KEY
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    reason TEXT NOT NULL,
    invalidated_at_utc TEXT NOT NULL
);

CREATE TRIGGER training_run_provenance_invalidations_identity_conflict
BEFORE INSERT ON training_run_provenance_invalidations
WHEN EXISTS (
    SELECT 1 FROM training_run_provenance_invalidations
    WHERE training_run_id=NEW.training_run_id
)
BEGIN
    SELECT RAISE(ABORT, 'training run provenance invalidation identity is immutable');
END;

CREATE TRIGGER training_run_provenance_invalidations_immutable_update
BEFORE UPDATE ON training_run_provenance_invalidations
BEGIN
    SELECT RAISE(ABORT, 'training run provenance invalidations are immutable');
END;

CREATE TRIGGER training_run_provenance_invalidations_immutable_delete
BEFORE DELETE ON training_run_provenance_invalidations
BEGIN
    SELECT RAISE(ABORT, 'training run provenance invalidations are immutable');
END;

CREATE VIEW valid_training_runs AS
SELECT training_runs.*
FROM training_runs
LEFT JOIN training_run_provenance_invalidations AS invalidation
    ON invalidation.training_run_id = training_runs.training_run_id
WHERE invalidation.training_run_id IS NULL;

-- Replace the migration-004 trigger: it only fires on `UPDATE OF
-- training_run_id`, which SQLite does not consider a match when the same
-- underlying column is updated through the rowid/_rowid_/oid aliases.
DROP TRIGGER training_runs_immutable_primary_key;

CREATE TRIGGER training_runs_immutable_identity
BEFORE UPDATE ON training_runs
WHEN NEW.training_run_id IS NOT OLD.training_run_id
BEGIN
    SELECT RAISE(ABORT, 'training run identity is immutable');
END;

-- Deterministically invalidate every training run that existed before 005:
-- its evidence was mutable under the old schema and cannot be proven now.
INSERT INTO training_run_provenance_invalidations (
    training_run_id, reason, invalidated_at_utc
)
SELECT training_run_id, 'pre005_mutable_provenance', datetime('now')
FROM training_runs;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_migration_005_identity.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run migrations 002/003/004 regression suites**

Run: `python -m pytest tests/database/test_migrations_v17.py tests/database/test_model_integrity_migration_v17.py -v`
Expected: PASS (untouched — 005 hasn't been added to the default packaged catalog yet in these tests' fixtures, since they build explicit 001-004 catalogs)

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/sql/005_provenance_integrity.sql tests/database/test_migration_005_identity.py
git commit -m "feat: migration 005 part A - rowid-safe identity guard and invalidation ledger"
```

---

## Task 5: Migration 005 Part B — training run and evaluation immutability/state machine

**Files:**
- Modify: `swmm_resilience/database/sql/005_provenance_integrity.sql` (append)
- Test: `tests/database/test_migration_005_lifecycle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_005_lifecycle.py
from pathlib import Path
import shutil

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def _catalog_through_005(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql",
        "002_model_integrity.sql",
        "003_model_integrity_guards.sql",
        "004_training_run_identity.sql",
        "005_provenance_integrity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    return catalog


def _insert_training_run(conn, training_run_id=1, status="PENDING"):
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (?, 'system', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', ?)
        """,
        (training_run_id, "a" * 64, status),
    )


def test_training_run_config_is_immutable(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _insert_training_run(conn)
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("UPDATE training_runs SET random_seed = 7 WHERE training_run_id = 1")
    conn.rollback()


def test_training_run_status_transitions(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _insert_training_run(conn, status="PENDING")
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "UPDATE training_runs SET status = 'COMPLETE' WHERE training_run_id = 1"
        )
    conn.rollback()

    conn.execute("UPDATE training_runs SET status = 'RUNNING' WHERE training_run_id = 1")
    conn.commit()
    with pytest.raises(Exception):
        conn.execute("UPDATE training_runs SET status = 'PENDING' WHERE training_run_id = 1")
    conn.rollback()


def test_training_run_cannot_be_deleted(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _insert_training_run(conn)
    conn.commit()
    with pytest.raises(Exception):
        conn.execute("DELETE FROM training_runs WHERE training_run_id = 1")
    conn.rollback()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_migration_005_lifecycle.py -v`
Expected: FAIL — no exception raised (updates currently succeed)

- [ ] **Step 3: Append Part B to the migration**

```sql
CREATE TRIGGER training_runs_immutable_configuration
BEFORE UPDATE ON training_runs
WHEN NEW.target IS NOT OLD.target
   OR NEW.feature_contract_id IS NOT OLD.feature_contract_id
   OR NEW.feature_contract_sha256 IS NOT OLD.feature_contract_sha256
   OR NEW.query_sql IS NOT OLD.query_sql
   OR NEW.query_params_json IS NOT OLD.query_params_json
   OR NEW.included_run_ids_json IS NOT OLD.included_run_ids_json
   OR NEW.grouping_strategy IS NOT OLD.grouping_strategy
   OR NEW.fold_count IS NOT OLD.fold_count
   OR NEW.random_seed IS NOT OLD.random_seed
   OR NEW.primary_metric IS NOT OLD.primary_metric
   OR NEW.tie_breakers_json IS NOT OLD.tie_breakers_json
   OR NEW.python_version IS NOT OLD.python_version
   OR NEW.library_versions_json IS NOT OLD.library_versions_json
BEGIN
    SELECT RAISE(ABORT, 'training run fitting configuration is immutable');
END;

CREATE TRIGGER training_runs_valid_status_transition
BEFORE UPDATE OF status ON training_runs
WHEN NOT (
    (OLD.status='PENDING' AND NEW.status IN ('RUNNING','FAILED'))
    OR (OLD.status='RUNNING' AND NEW.status IN ('COMPLETE','FAILED'))
)
BEGIN
    SELECT RAISE(ABORT, 'invalid training run status transition');
END;

CREATE TRIGGER training_runs_no_delete
BEFORE DELETE ON training_runs
BEGIN
    SELECT RAISE(ABORT, 'training runs cannot be deleted');
END;

CREATE TRIGGER model_evaluations_immutable_identity
BEFORE UPDATE ON model_evaluations
WHEN NEW.evaluation_id IS NOT OLD.evaluation_id
   OR NEW.training_run_id IS NOT OLD.training_run_id
   OR NEW.task IS NOT OLD.task
   OR NEW.algorithm IS NOT OLD.algorithm
   OR NEW.hyperparameters_json IS NOT OLD.hyperparameters_json
   OR NEW.fold_id IS NOT OLD.fold_id
   OR NEW.train_run_ids_json IS NOT OLD.train_run_ids_json
   OR NEW.validation_run_ids_json IS NOT OLD.validation_run_ids_json
BEGIN
    SELECT RAISE(ABORT, 'model evaluation identity/configuration is immutable');
END;

CREATE TRIGGER model_evaluations_valid_status_transition
BEFORE UPDATE OF status ON model_evaluations
WHEN NOT (
    (OLD.status='PENDING' AND NEW.status IN ('RUNNING','FAILED'))
    OR (OLD.status='RUNNING' AND NEW.status IN ('COMPLETE','FAILED'))
)
BEGIN
    SELECT RAISE(ABORT, 'invalid model evaluation status transition');
END;

CREATE TRIGGER model_evaluations_no_delete
BEFORE DELETE ON model_evaluations
BEGIN
    SELECT RAISE(ABORT, 'model evaluations cannot be deleted');
END;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_migration_005_lifecycle.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Re-run Task 4's tests to confirm no regression**

Run: `python -m pytest tests/database/test_migration_005_identity.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/sql/005_provenance_integrity.sql tests/database/test_migration_005_lifecycle.py
git commit -m "feat: migration 005 part B - training run and evaluation state machines"
```

---

## Task 6: Migration 005 Part C — normalized membership and PENDING/RUNNING consistency boundary

**Files:**
- Modify: `swmm_resilience/database/sql/005_provenance_integrity.sql` (append)
- Test: `tests/database/test_migration_005_membership.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_005_membership.py
from pathlib import Path
import shutil

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def _catalog_through_005(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql",
        "002_model_integrity.sql",
        "003_model_integrity_guards.sql",
        "004_training_run_identity.sql",
        "005_provenance_integrity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    return catalog


def _seed_network_and_runs(conn, n=2):
    conn.execute(
        """
        INSERT INTO networks (network_id, network_sha256, name, source_filename,
                               inp_bytes, flow_units, created_at_utc)
        VALUES (1, ?, 'net', 'net.inp', ?, 'LPS', '2026-08-22T00:00:00+00:00')
        """,
        ("f" * 64, b"inp-bytes"),
    )
    for run_id in range(1, n + 1):
        conn.execute(
            """
            INSERT INTO runs (run_id, network_id, scenario_id, status, node_count)
            VALUES (?, 1, NULL, 'COMPLETE', 1)
            """,
            (run_id,),
        )


def _insert_training_run(conn, training_run_id, included_run_ids, status="PENDING"):
    import json

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (?, 'system', 'tabular_v3_17', ?, 'SELECT 1', '{}', ?,
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', ?)
        """,
        (training_run_id, "a" * 64, json.dumps(sorted(included_run_ids)), status),
    )


def test_membership_insert_requires_pending_owner(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_network_and_runs(conn)
    _insert_training_run(conn, 1, [1], status="RUNNING")
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
        )
    conn.rollback()


def test_membership_insert_must_be_in_canonical_json(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_network_and_runs(conn)
    _insert_training_run(conn, 1, [1], status="PENDING")
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 2)"
        )
    conn.rollback()

    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
    )
    conn.commit()


def test_membership_rows_are_immutable(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_network_and_runs(conn)
    _insert_training_run(conn, 1, [1], status="PENDING")
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("DELETE FROM training_run_inputs WHERE training_run_id = 1")
    conn.rollback()


def test_running_requires_complete_and_equal_membership(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_network_and_runs(conn, n=2)
    _insert_training_run(conn, 1, [1, 2], status="PENDING")
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("UPDATE training_runs SET status = 'RUNNING' WHERE training_run_id = 1")
    conn.rollback()

    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 2)"
    )
    conn.execute("UPDATE training_runs SET status = 'RUNNING' WHERE training_run_id = 1")
    conn.commit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_migration_005_membership.py -v`
Expected: FAIL — `sqlite3.OperationalError: no such table: training_run_inputs`

- [ ] **Step 3: Append Part C to the migration**

```sql
CREATE TABLE training_run_inputs (
    training_run_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    PRIMARY KEY (training_run_id, run_id),
    FOREIGN KEY (training_run_id)
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    FOREIGN KEY (run_id)
        REFERENCES runs(run_id) ON DELETE RESTRICT
);

CREATE INDEX idx_training_run_inputs_reverse
    ON training_run_inputs(run_id, training_run_id);

CREATE TABLE model_evaluation_runs (
    evaluation_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('train','validation')),
    run_id INTEGER NOT NULL,
    PRIMARY KEY (evaluation_id, role, run_id),
    FOREIGN KEY (evaluation_id)
        REFERENCES model_evaluations(evaluation_id) ON DELETE RESTRICT,
    FOREIGN KEY (run_id)
        REFERENCES runs(run_id) ON DELETE RESTRICT
);

CREATE INDEX idx_model_evaluation_runs_reverse
    ON model_evaluation_runs(run_id, evaluation_id, role);

CREATE TRIGGER training_run_inputs_no_update
BEFORE UPDATE ON training_run_inputs
BEGIN
    SELECT RAISE(ABORT, 'training run membership is immutable');
END;

CREATE TRIGGER training_run_inputs_no_delete
BEFORE DELETE ON training_run_inputs
BEGIN
    SELECT RAISE(ABORT, 'training run membership is immutable');
END;

CREATE TRIGGER training_run_inputs_owner_pending
BEFORE INSERT ON training_run_inputs
WHEN NOT EXISTS (
    SELECT 1 FROM training_runs
    WHERE training_run_id=NEW.training_run_id AND status='PENDING'
)
BEGIN
    SELECT RAISE(ABORT, 'training run membership requires a PENDING owner');
END;

CREATE TRIGGER training_run_inputs_within_canonical_json
BEFORE INSERT ON training_run_inputs
WHEN NOT EXISTS (
    SELECT 1
    FROM training_runs, json_each(training_runs.included_run_ids_json)
    WHERE training_runs.training_run_id=NEW.training_run_id
      AND json_each.value=NEW.run_id
)
BEGIN
    SELECT RAISE(ABORT, 'training run membership must be listed in included_run_ids_json');
END;

CREATE TRIGGER model_evaluation_runs_no_update
BEFORE UPDATE ON model_evaluation_runs
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership is immutable');
END;

CREATE TRIGGER model_evaluation_runs_no_delete
BEFORE DELETE ON model_evaluation_runs
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership is immutable');
END;

CREATE TRIGGER model_evaluation_runs_owner_pending
BEFORE INSERT ON model_evaluation_runs
WHEN NOT EXISTS (
    SELECT 1 FROM model_evaluations
    WHERE evaluation_id=NEW.evaluation_id AND status='PENDING'
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership requires a PENDING owner');
END;

CREATE TRIGGER model_evaluation_runs_within_canonical_json
BEFORE INSERT ON model_evaluation_runs
WHEN NOT EXISTS (
    SELECT 1
    FROM model_evaluations, json_each(
        CASE NEW.role
            WHEN 'train' THEN model_evaluations.train_run_ids_json
            ELSE model_evaluations.validation_run_ids_json
        END
    )
    WHERE model_evaluations.evaluation_id=NEW.evaluation_id
      AND json_each.value=NEW.run_id
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership must be listed in its canonical JSON array');
END;

CREATE TRIGGER model_evaluation_runs_within_training_run
BEFORE INSERT ON model_evaluation_runs
WHEN NOT EXISTS (
    SELECT 1
    FROM model_evaluations
    JOIN training_run_inputs
        ON training_run_inputs.training_run_id = model_evaluations.training_run_id
       AND training_run_inputs.run_id = NEW.run_id
    WHERE model_evaluations.evaluation_id = NEW.evaluation_id
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership must be contained in the training run membership');
END;

-- Consistency boundary: PENDING -> RUNNING requires normalized membership
-- to be non-empty and exactly equal to the canonical JSON array.
CREATE TRIGGER training_runs_running_requires_complete_membership
BEFORE UPDATE OF status ON training_runs
WHEN NEW.status='RUNNING' AND OLD.status='PENDING'
  AND (
    NOT EXISTS (
        SELECT 1 FROM training_run_inputs WHERE training_run_id=OLD.training_run_id
    )
    OR (
        SELECT COUNT(*) FROM training_run_inputs
        WHERE training_run_id=OLD.training_run_id
    ) <> (SELECT COUNT(*) FROM json_each(OLD.included_run_ids_json))
    OR EXISTS (
        SELECT value FROM json_each(OLD.included_run_ids_json)
        WHERE value NOT IN (
            SELECT run_id FROM training_run_inputs
            WHERE training_run_id=OLD.training_run_id
        )
    )
  )
BEGIN
    SELECT RAISE(
        ABORT,
        'training run membership must be non-empty and equal to included_run_ids_json before RUNNING'
    );
END;

CREATE TRIGGER model_evaluations_running_requires_complete_membership
BEFORE UPDATE OF status ON model_evaluations
WHEN NEW.status='RUNNING' AND OLD.status='PENDING'
  AND (
    NOT EXISTS (
        SELECT 1 FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='train'
    )
    OR NOT EXISTS (
        SELECT 1 FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='validation'
    )
    OR EXISTS (
        SELECT run_id FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='train'
        INTERSECT
        SELECT run_id FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='validation'
    )
    OR (
        SELECT COUNT(*) FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='train'
    ) <> (SELECT COUNT(*) FROM json_each(OLD.train_run_ids_json))
    OR (
        SELECT COUNT(*) FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='validation'
    ) <> (SELECT COUNT(*) FROM json_each(OLD.validation_run_ids_json))
    OR OLD.fold_id >= (
        SELECT fold_count FROM training_runs
        WHERE training_run_id=OLD.training_run_id
    )
  )
BEGIN
    SELECT RAISE(
        ABORT,
        'evaluation membership must be complete, disjoint, and equal to its JSON arrays before RUNNING'
    );
END;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_migration_005_membership.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Re-run Tasks 4-5 tests**

Run: `python -m pytest tests/database/test_migration_005_identity.py tests/database/test_migration_005_lifecycle.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/sql/005_provenance_integrity.sql tests/database/test_migration_005_membership.py
git commit -m "feat: migration 005 part C - normalized membership and RUNNING consistency boundary"
```

---

## Task 7: Migration 005 Part D — append-only OOF evidence with validation

**Files:**
- Modify: `swmm_resilience/database/sql/005_provenance_integrity.sql` (append)
- Test: `tests/database/test_migration_005_oof.py`

**Note:** `oof_predictions` (from 001) has no `fold_id` uniqueness beyond `PRIMARY KEY(evaluation_id, run_id, node_pk)` and no `probability`/domain checks beyond the base column types. This task adds the insert-time validation trigger describing the checks in spec 3.4, and makes the table append-only (no `UPDATE`/`DELETE`/`REPLACE`).

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_005_oof.py
from pathlib import Path
import shutil

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def _catalog_through_005(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql",
        "002_model_integrity.sql",
        "003_model_integrity_guards.sql",
        "004_training_run_identity.sql",
        "005_provenance_integrity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    return catalog


def _seed(conn):
    import json

    conn.execute(
        """
        INSERT INTO networks (network_id, network_sha256, name, source_filename,
                               inp_bytes, flow_units, created_at_utc)
        VALUES (1, ?, 'net', 'net.inp', ?, 'LPS', '2026-08-22T00:00:00+00:00')
        """,
        ("f" * 64, b"inp-bytes"),
    )
    conn.execute(
        "INSERT INTO nodes (node_pk, network_id, node_uid) VALUES (1, 1, 'N1')"
    )
    conn.execute(
        """
        INSERT INTO runs (run_id, network_id, scenario_id, status, node_count)
        VALUES (1, 1, NULL, 'COMPLETE', 1)
        """
    )
    conn.execute(
        """
        INSERT INTO node_results (run_id, node_pk, inunda, vol_inundacion_m3)
        VALUES (1, 1, 1, 3.5)
        """
    )
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[1]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'PENDING')
        """,
        ("a" * 64,),
    )
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
    )
    conn.execute("UPDATE training_runs SET status='RUNNING' WHERE training_run_id=1")
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (1, 1, 'classification', 'xgboost', '{}', 0, '[]', '[1]',
                  'PENDING', 0, 0)
        """
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'validation', 1)"
    )
    conn.execute("UPDATE model_evaluations SET status='RUNNING' WHERE evaluation_id=1")
    conn.commit()


def test_oof_insert_requires_running_owner_and_matches_persisted_target(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed(conn)

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO oof_predictions (
                evaluation_id, run_id, node_pk, target, observed, predicted,
                probability, fold_id
            ) VALUES (1, 1, 1, 'inunda', 0, 1, 0.9, 0)
            """
        )
    conn.rollback()

    conn.execute(
        """
        INSERT INTO oof_predictions (
            evaluation_id, run_id, node_pk, target, observed, predicted,
            probability, fold_id
        ) VALUES (1, 1, 1, 'inunda', 1, 1, 0.9, 0)
        """
    )
    conn.commit()


def test_oof_rows_are_append_only(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed(conn)
    conn.execute(
        """
        INSERT INTO oof_predictions (
            evaluation_id, run_id, node_pk, target, observed, predicted,
            probability, fold_id
        ) VALUES (1, 1, 1, 'inunda', 1, 1, 0.9, 0)
        """
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "UPDATE oof_predictions SET predicted = 0 WHERE evaluation_id = 1"
        )
    conn.rollback()

    with pytest.raises(Exception):
        conn.execute("DELETE FROM oof_predictions WHERE evaluation_id = 1")
    conn.rollback()

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT OR REPLACE INTO oof_predictions (
                evaluation_id, run_id, node_pk, target, observed, predicted,
                probability, fold_id
            ) VALUES (1, 1, 1, 'inunda', 1, 0, 0.1, 0)
            """
        )
    conn.rollback()


def test_oof_rejects_out_of_domain_classification_values(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed(conn)

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO oof_predictions (
                evaluation_id, run_id, node_pk, target, observed, predicted,
                probability, fold_id
            ) VALUES (1, 1, 1, 'inunda', 1, 1, 1.5, 0)
            """
        )
    conn.rollback()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_migration_005_oof.py -v`
Expected: FAIL — first insert (mismatched `observed`) currently succeeds instead of raising

- [ ] **Step 3: Append Part D to the migration**

```sql
CREATE TRIGGER oof_predictions_no_update
BEFORE UPDATE ON oof_predictions
BEGIN
    SELECT RAISE(ABORT, 'OOF predictions are append-only');
END;

CREATE TRIGGER oof_predictions_no_delete
BEFORE DELETE ON oof_predictions
BEGIN
    SELECT RAISE(ABORT, 'OOF predictions are append-only');
END;

CREATE TRIGGER oof_predictions_identity_conflict
BEFORE INSERT ON oof_predictions
WHEN EXISTS (
    SELECT 1 FROM oof_predictions
    WHERE evaluation_id=NEW.evaluation_id AND run_id=NEW.run_id AND node_pk=NEW.node_pk
)
BEGIN
    SELECT RAISE(ABORT, 'OOF prediction identity is immutable');
END;

CREATE TRIGGER oof_predictions_validate_insert
BEFORE INSERT ON oof_predictions
WHEN NOT EXISTS (
    SELECT 1
    FROM model_evaluations
    JOIN training_runs
        ON training_runs.training_run_id = model_evaluations.training_run_id
    WHERE model_evaluations.evaluation_id = NEW.evaluation_id
      AND model_evaluations.status = 'RUNNING'
      AND training_runs.status = 'RUNNING'
      AND model_evaluations.fold_id = NEW.fold_id
      AND (
          (NEW.target = 'inunda' AND model_evaluations.task = 'classification')
          OR (NEW.target = 'vol_inundacion_m3' AND model_evaluations.task = 'regression')
      )
)
   OR NOT EXISTS (
       SELECT 1 FROM model_evaluation_runs
       WHERE evaluation_id = NEW.evaluation_id
         AND role = 'validation'
         AND run_id = NEW.run_id
   )
   OR NOT EXISTS (
       SELECT 1 FROM node_results
       WHERE run_id = NEW.run_id AND node_pk = NEW.node_pk
   )
   OR NEW.observed <> (
       SELECT CASE NEW.target
           WHEN 'inunda' THEN node_results.inunda
           ELSE node_results.vol_inundacion_m3
       END
       FROM node_results
       WHERE node_results.run_id = NEW.run_id AND node_results.node_pk = NEW.node_pk
   )
   OR NEW.predicted IS NULL OR NEW.predicted <> NEW.predicted  -- rejects NaN (NaN <> NaN is true)
   OR (
       NEW.target = 'inunda' AND (
           NEW.observed NOT IN (0, 1)
           OR NEW.predicted NOT IN (0, 1)
           OR NEW.probability IS NULL
           OR NEW.probability < 0 OR NEW.probability > 1
       )
   )
   OR (
       NEW.target = 'vol_inundacion_m3' AND (
           NEW.observed < 0
           OR NEW.probability IS NOT NULL
       )
   )
BEGIN
    SELECT RAISE(ABORT, 'OOF prediction fails provenance/domain validation');
END;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_migration_005_oof.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Re-run Tasks 4-6 tests**

Run: `python -m pytest tests/database/test_migration_005_identity.py tests/database/test_migration_005_lifecycle.py tests/database/test_migration_005_membership.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/sql/005_provenance_integrity.sql tests/database/test_migration_005_oof.py
git commit -m "feat: migration 005 part D - append-only validated OOF evidence"
```

---

## Task 8: Migration 005 Part E — candidate, ranking, and promotion evidence chain

**Files:**
- Modify: `swmm_resilience/database/sql/005_provenance_integrity.sql` (append)
- Test: `tests/database/test_migration_005_candidates.py`

This is the largest remaining piece (spec 3.4.1 and 3.5). It adds the entities that connect completed evaluations to a ranking-justified promotion.

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_005_candidates.py
from pathlib import Path
import shutil

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def _catalog_through_005(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql",
        "002_model_integrity.sql",
        "003_model_integrity_guards.sql",
        "004_training_run_identity.sql",
        "005_provenance_integrity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    return catalog


def test_candidate_tables_exist_and_are_append_only(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    for table in (
        "model_candidates",
        "model_candidate_evaluations",
        "model_candidate_finalizations",
        "model_artifact_candidates",
        "model_rankings",
        "model_ranking_entries",
        "model_ranking_scores",
        "model_ranking_finalizations",
        "model_promotion_rankings",
        "model_promotion_finalizations",
    ):
        conn.execute(f"SELECT * FROM {table} LIMIT 0")  # raises if table is missing


def test_candidate_evaluation_link_requires_matching_task_and_algorithm(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'RUNNING')
        """,
        ("a" * 64,),
    )
    conn.execute(
        """
        INSERT INTO model_candidates (
            candidate_id, training_run_id, task, algorithm, hyperparameters_json,
            preprocessing_json, feature_contract_id, feature_contract_sha256,
            ordered_features_json, target_transform_json, pipeline_version,
            candidate_definition_sha256
        ) VALUES (1, 1, 'classification', 'xgboost', '{}', '{}', 'tabular_v3_17',
                  ?, '[]', '{}', 'v1', ?)
        """,
        ("a" * 64, "c" * 64),
    )
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (1, 1, 'regression', 'xgboost', '{}', 0, '[]', '[]', 'PENDING', 0, 0)
        """
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (1, 1)"
        )
    conn.rollback()


def test_evaluation_id_is_unique_across_candidate_links(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'RUNNING')
        """,
        ("a" * 64,),
    )
    for candidate_id in (1, 2):
        conn.execute(
            """
            INSERT INTO model_candidates (
                candidate_id, training_run_id, task, algorithm, hyperparameters_json,
                preprocessing_json, feature_contract_id, feature_contract_sha256,
                ordered_features_json, target_transform_json, pipeline_version,
                candidate_definition_sha256
            ) VALUES (?, 1, 'classification', 'xgboost', '{}', '{}', 'tabular_v3_17',
                      ?, '[]', '{}', 'v1', ?)
            """,
            (candidate_id, "a" * 64, f"{candidate_id:064d}"),
        )
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (1, 1, 'classification', 'xgboost', '{}', 0, '[]', '[]', 'PENDING', 0, 0)
        """
    )
    conn.execute(
        "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (1, 1)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (2, 1)"
        )
    conn.rollback()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_migration_005_candidates.py -v`
Expected: FAIL — `sqlite3.OperationalError: no such table: model_candidates`

- [ ] **Step 3: Append Part E to the migration**

```sql
CREATE TABLE model_candidates (
    candidate_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    task TEXT NOT NULL CHECK(task IN ('classification','regression')),
    algorithm TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    preprocessing_json TEXT NOT NULL,
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v3_17'),
    feature_contract_sha256 TEXT NOT NULL CHECK(length(feature_contract_sha256)=64),
    ordered_features_json TEXT NOT NULL,
    target_transform_json TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    candidate_definition_sha256 TEXT NOT NULL CHECK(length(candidate_definition_sha256)=64)
);

CREATE INDEX idx_model_candidates_training_run
    ON model_candidates(training_run_id, task, algorithm);

CREATE TRIGGER model_candidates_no_update
BEFORE UPDATE ON model_candidates
BEGIN
    SELECT RAISE(ABORT, 'model candidates are immutable');
END;

CREATE TRIGGER model_candidates_no_delete
BEFORE DELETE ON model_candidates
BEGIN
    SELECT RAISE(ABORT, 'model candidates are immutable');
END;

CREATE TRIGGER model_candidates_owner_running
BEFORE INSERT ON model_candidates
WHEN NOT EXISTS (
    SELECT 1 FROM training_runs
    WHERE training_run_id=NEW.training_run_id AND status='RUNNING'
)
BEGIN
    SELECT RAISE(ABORT, 'candidates are inserted only while the training run is RUNNING');
END;

CREATE TABLE model_candidate_evaluations (
    candidate_id INTEGER NOT NULL
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT,
    evaluation_id INTEGER NOT NULL UNIQUE
        REFERENCES model_evaluations(evaluation_id) ON DELETE RESTRICT,
    PRIMARY KEY (candidate_id, evaluation_id)
);

CREATE TRIGGER model_candidate_evaluations_no_update
BEFORE UPDATE ON model_candidate_evaluations
BEGIN
    SELECT RAISE(ABORT, 'candidate/evaluation links are immutable');
END;

CREATE TRIGGER model_candidate_evaluations_no_delete
BEFORE DELETE ON model_candidate_evaluations
BEGIN
    SELECT RAISE(ABORT, 'candidate/evaluation links are immutable');
END;

CREATE TRIGGER model_candidate_evaluations_matches_candidate
BEFORE INSERT ON model_candidate_evaluations
WHEN NOT EXISTS (
    SELECT 1
    FROM model_candidates
    JOIN model_evaluations
        ON model_evaluations.training_run_id = model_candidates.training_run_id
       AND model_evaluations.task = model_candidates.task
       AND model_evaluations.algorithm = model_candidates.algorithm
       AND model_evaluations.hyperparameters_json = model_candidates.hyperparameters_json
    WHERE model_candidates.candidate_id = NEW.candidate_id
      AND model_evaluations.evaluation_id = NEW.evaluation_id
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation does not match its candidate task/algorithm/hyperparameters');
END;

CREATE TRIGGER model_candidate_evaluations_not_after_finalization
BEFORE INSERT ON model_candidate_evaluations
WHEN EXISTS (
    SELECT 1 FROM model_candidate_finalizations WHERE candidate_id=NEW.candidate_id
)
BEGIN
    SELECT RAISE(ABORT, 'no evaluation link is added after candidate finalization');
END;

CREATE TABLE model_candidate_finalizations (
    candidate_id INTEGER PRIMARY KEY
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT,
    finalized_at_utc TEXT NOT NULL
);

CREATE TRIGGER model_candidate_finalizations_no_update
BEFORE UPDATE ON model_candidate_finalizations
BEGIN
    SELECT RAISE(ABORT, 'candidate finalizations are immutable');
END;

CREATE TRIGGER model_candidate_finalizations_no_delete
BEFORE DELETE ON model_candidate_finalizations
BEGIN
    SELECT RAISE(ABORT, 'candidate finalizations are immutable');
END;

-- One COMPLETE evaluation per fold 0..fold_count-1; full, disjoint-per-fold,
-- population-covering validation membership; every expected (run_id,node_pk)
-- has exactly one OOF row.
CREATE TRIGGER model_candidate_finalizations_validate_insert
BEFORE INSERT ON model_candidate_finalizations
WHEN (
    SELECT COUNT(*)
    FROM model_candidate_evaluations AS link
    JOIN model_evaluations AS evaluation
        ON evaluation.evaluation_id = link.evaluation_id
    WHERE link.candidate_id = NEW.candidate_id AND evaluation.status = 'COMPLETE'
) <> (
    SELECT fold_count FROM training_runs
    WHERE training_run_id = (
        SELECT training_run_id FROM model_candidates WHERE candidate_id = NEW.candidate_id
    )
)
   OR EXISTS (
       SELECT 1
       FROM model_candidate_evaluations AS link
       JOIN model_evaluations AS evaluation
           ON evaluation.evaluation_id = link.evaluation_id
       WHERE link.candidate_id = NEW.candidate_id AND evaluation.status <> 'COMPLETE'
   )
   OR EXISTS (
       -- every validation (run_id,node_pk) for this candidate's runs must have
       -- exactly one OOF row per linked evaluation
       SELECT link.evaluation_id
       FROM model_candidate_evaluations AS link
       JOIN model_evaluation_runs AS membership
           ON membership.evaluation_id = link.evaluation_id AND membership.role = 'validation'
       JOIN node_results AS result
           ON result.run_id = membership.run_id
       LEFT JOIN oof_predictions AS oof
           ON oof.evaluation_id = link.evaluation_id
          AND oof.run_id = result.run_id
          AND oof.node_pk = result.node_pk
       WHERE link.candidate_id = NEW.candidate_id
         AND oof.evaluation_id IS NULL
   )
BEGIN
    SELECT RAISE(
        ABORT,
        'candidate finalization requires one COMPLETE evaluation per fold with full OOF coverage'
    );
END;

CREATE TABLE model_artifact_candidates (
    model_id INTEGER PRIMARY KEY
        REFERENCES trained_models(model_id) ON DELETE RESTRICT,
    candidate_id INTEGER NOT NULL
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT
);

CREATE TRIGGER model_artifact_candidates_no_update
BEFORE UPDATE ON model_artifact_candidates
BEGIN
    SELECT RAISE(ABORT, 'artifact/candidate links are immutable');
END;

CREATE TRIGGER model_artifact_candidates_no_delete
BEFORE DELETE ON model_artifact_candidates
BEGIN
    SELECT RAISE(ABORT, 'artifact/candidate links are immutable');
END;

CREATE TRIGGER model_artifact_candidates_requires_finalized_matching_candidate
BEFORE INSERT ON model_artifact_candidates
WHEN NOT EXISTS (
    SELECT 1
    FROM model_candidates AS candidate
    JOIN model_candidate_finalizations AS finalization
        ON finalization.candidate_id = candidate.candidate_id
    JOIN trained_models AS artifact
        ON artifact.training_run_id = candidate.training_run_id
       AND artifact.feature_contract_id = candidate.feature_contract_id
       AND artifact.feature_contract_sha256 = candidate.feature_contract_sha256
       AND artifact.ordered_features_json = candidate.ordered_features_json
       AND artifact.preprocessing_json = candidate.preprocessing_json
       AND artifact.target_transform_json = candidate.target_transform_json
       AND artifact.hyperparameters_json = candidate.hyperparameters_json
       AND artifact.algorithm = candidate.algorithm
    WHERE candidate.candidate_id = NEW.candidate_id
      AND artifact.model_id = NEW.model_id
)
BEGIN
    SELECT RAISE(
        ABORT,
        'artifact link requires a finalized candidate whose recipe matches the artifact'
    );
END;

CREATE TABLE model_rankings (
    ranking_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3','system')),
    primary_metric TEXT NOT NULL,
    primary_direction TEXT NOT NULL CHECK(primary_direction IN ('maximize','minimize')),
    metric_registry_id TEXT NOT NULL,
    metric_registry_sha256 TEXT NOT NULL CHECK(length(metric_registry_sha256)=64),
    tie_breakers_json TEXT NOT NULL,
    invalid_score_policy TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TRIGGER model_rankings_no_update
BEFORE UPDATE ON model_rankings
BEGIN
    SELECT RAISE(ABORT, 'model rankings are immutable');
END;

CREATE TRIGGER model_rankings_no_delete
BEFORE DELETE ON model_rankings
BEGIN
    SELECT RAISE(ABORT, 'model rankings are immutable');
END;

CREATE TABLE model_ranking_entries (
    ranking_id INTEGER NOT NULL
        REFERENCES model_rankings(ranking_id) ON DELETE RESTRICT,
    entry_id INTEGER NOT NULL,
    classifier_candidate_id INTEGER
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT,
    regressor_candidate_id INTEGER
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT,
    PRIMARY KEY (ranking_id, entry_id),
    CHECK (classifier_candidate_id IS NOT NULL OR regressor_candidate_id IS NOT NULL)
);

CREATE TRIGGER model_ranking_entries_no_update
BEFORE UPDATE ON model_ranking_entries
BEGIN
    SELECT RAISE(ABORT, 'ranking entries are immutable');
END;

CREATE TRIGGER model_ranking_entries_no_delete
BEFORE DELETE ON model_ranking_entries
BEGIN
    SELECT RAISE(ABORT, 'ranking entries are immutable');
END;

CREATE TRIGGER model_ranking_entries_not_after_finalization
BEFORE INSERT ON model_ranking_entries
WHEN EXISTS (
    SELECT 1 FROM model_ranking_finalizations WHERE ranking_id=NEW.ranking_id
)
BEGIN
    SELECT RAISE(ABORT, 'no entry is added after ranking finalization');
END;

CREATE TABLE model_ranking_scores (
    ranking_id INTEGER NOT NULL,
    entry_id INTEGER NOT NULL,
    metric_ordinal INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL,
    valid INTEGER NOT NULL CHECK(valid IN (0,1)),
    invalid_reason TEXT,
    PRIMARY KEY (ranking_id, entry_id, metric_ordinal),
    FOREIGN KEY (ranking_id, entry_id)
        REFERENCES model_ranking_entries(ranking_id, entry_id) ON DELETE RESTRICT,
    CHECK ((valid=1 AND value IS NOT NULL) OR (valid=0 AND invalid_reason IS NOT NULL))
);

CREATE TRIGGER model_ranking_scores_no_update
BEFORE UPDATE ON model_ranking_scores
BEGIN
    SELECT RAISE(ABORT, 'ranking scores are immutable');
END;

CREATE TRIGGER model_ranking_scores_no_delete
BEFORE DELETE ON model_ranking_scores
BEGIN
    SELECT RAISE(ABORT, 'ranking scores are immutable');
END;

CREATE TRIGGER model_ranking_scores_not_after_finalization
BEFORE INSERT ON model_ranking_scores
WHEN EXISTS (
    SELECT 1 FROM model_ranking_finalizations WHERE ranking_id=NEW.ranking_id
)
BEGIN
    SELECT RAISE(ABORT, 'no score is added after ranking finalization');
END;

CREATE TABLE model_ranking_finalizations (
    ranking_id INTEGER PRIMARY KEY
        REFERENCES model_rankings(ranking_id) ON DELETE RESTRICT,
    winner_entry_id INTEGER NOT NULL,
    finalized_at_utc TEXT NOT NULL,
    FOREIGN KEY (ranking_id, winner_entry_id)
        REFERENCES model_ranking_entries(ranking_id, entry_id) ON DELETE RESTRICT
);

CREATE TRIGGER model_ranking_finalizations_no_update
BEFORE UPDATE ON model_ranking_finalizations
BEGIN
    SELECT RAISE(ABORT, 'ranking finalizations are immutable');
END;

CREATE TRIGGER model_ranking_finalizations_no_delete
BEFORE DELETE ON model_ranking_finalizations
BEGIN
    SELECT RAISE(ABORT, 'ranking finalizations are immutable');
END;

CREATE TABLE model_promotion_rankings (
    promotion_id INTEGER PRIMARY KEY
        REFERENCES model_promotions(promotion_id) ON DELETE RESTRICT,
    ranking_id INTEGER NOT NULL
        REFERENCES model_rankings(ranking_id) ON DELETE RESTRICT
);

CREATE TRIGGER model_promotion_rankings_no_update
BEFORE UPDATE ON model_promotion_rankings
BEGIN
    SELECT RAISE(ABORT, 'promotion/ranking links are immutable');
END;

CREATE TRIGGER model_promotion_rankings_no_delete
BEFORE DELETE ON model_promotion_rankings
BEGIN
    SELECT RAISE(ABORT, 'promotion/ranking links are immutable');
END;

CREATE TRIGGER model_promotion_rankings_requires_finalized_ranking
BEFORE INSERT ON model_promotion_rankings
WHEN NOT EXISTS (
    SELECT 1 FROM model_ranking_finalizations WHERE ranking_id=NEW.ranking_id
)
BEGIN
    SELECT RAISE(ABORT, 'promotion must link a finalized ranking');
END;

CREATE TABLE model_promotion_finalizations (
    promotion_id INTEGER PRIMARY KEY
        REFERENCES model_promotions(promotion_id) ON DELETE RESTRICT,
    finalized_at_utc TEXT NOT NULL
);

CREATE TRIGGER model_promotion_finalizations_no_update
BEFORE UPDATE ON model_promotion_finalizations
BEGIN
    SELECT RAISE(ABORT, 'promotion finalizations are immutable');
END;

CREATE TRIGGER model_promotion_finalizations_no_delete
BEFORE DELETE ON model_promotion_finalizations
BEGIN
    SELECT RAISE(ABORT, 'promotion finalizations are immutable');
END;

-- Promotion artifacts must link to the ranking's winner candidate(s), and
-- primary_metric/primary_value must equal the winner's valid primary score.
CREATE TRIGGER model_promotion_finalizations_validate_insert
BEFORE INSERT ON model_promotion_finalizations
WHEN NOT EXISTS (
    SELECT 1
    FROM model_promotion_rankings AS link
    JOIN model_ranking_finalizations AS finalization
        ON finalization.ranking_id = link.ranking_id
    JOIN model_ranking_scores AS score
        ON score.ranking_id = finalization.ranking_id
       AND score.entry_id = finalization.winner_entry_id
       AND score.metric_ordinal = 0
       AND score.valid = 1
    JOIN model_promotions AS promotion
        ON promotion.promotion_id = link.promotion_id
       AND promotion.primary_metric = (
           SELECT primary_metric FROM model_rankings WHERE ranking_id = finalization.ranking_id
       )
       AND promotion.primary_value = score.value
    JOIN model_ranking_entries AS entry
        ON entry.ranking_id = finalization.ranking_id
       AND entry.entry_id = finalization.winner_entry_id
    WHERE link.promotion_id = NEW.promotion_id
      AND (
          entry.classifier_candidate_id IS NULL
          OR EXISTS (
              SELECT 1 FROM model_artifact_candidates
              WHERE candidate_id = entry.classifier_candidate_id
                AND model_id = promotion.classifier_model_id
          )
      )
      AND (
          entry.regressor_candidate_id IS NULL
          OR EXISTS (
              SELECT 1 FROM model_artifact_candidates
              WHERE candidate_id = entry.regressor_candidate_id
                AND model_id = promotion.regressor_model_id
          )
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'promotion finalization requires its artifacts and metric to match the finalized ranking winner'
    );
END;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_migration_005_candidates.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Re-run all migration-005 tests so far**

Run: `python -m pytest tests/database/test_migration_005_identity.py tests/database/test_migration_005_lifecycle.py tests/database/test_migration_005_membership.py tests/database/test_migration_005_oof.py tests/database/test_migration_005_candidates.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/sql/005_provenance_integrity.sql tests/database/test_migration_005_candidates.py
git commit -m "feat: migration 005 part E - candidate, ranking, and promotion evidence chain"
```

---

## Task 9: Migration 005 Part F — legacy promotion invalidation and rebuilt views

**Files:**
- Modify: `swmm_resilience/database/sql/005_provenance_integrity.sql` (append — this closes out the migration file)
- Test: `tests/database/test_migration_005_legacy_promotions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_005_legacy_promotions.py
from pathlib import Path
import shutil

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def test_every_pre005_promotion_is_invalidated_and_selections_are_empty(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql",
        "002_model_integrity.sql",
        "003_model_integrity_guards.sql",
        "004_training_run_identity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status, completed_at_utc
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}',
                  'COMPLETE', '2026-08-21T00:00:00+00:00')
        """,
        ("a" * 64,),
    )
    conn.execute(
        """
        INSERT INTO trained_models (
            model_id, training_run_id, target, algorithm, hyperparameters_json,
            preprocessing_json, feature_contract_id, feature_contract_sha256,
            ordered_features_json, target_transform_json, query_params_json,
            included_run_ids_json, random_seed, grouping_strategy, python_version,
            library_versions_json, model_sha256, model_blob, created_at_utc
        ) VALUES (1, 1, 'inunda', 'xgboost', '{}', '{}', 'tabular_v3_17', ?, '[]',
                  '{}', '{}', '[]', 42, 'group_kfold', '3.11', '{}', ?, ?,
                  '2026-08-21T00:00:00+00:00')
        """,
        ("b" * 64, "c" * 64, b"blob"),
    )
    conn.execute(
        """
        INSERT INTO model_promotions (
            promotion_id, training_run_id, target, classifier_model_id,
            primary_metric, primary_value, tie_breakers_json, ranking_json,
            promoted_at_utc
        ) VALUES (1, 1, 'inunda', 1, 'roc_auc', 0.9, '[]', '{}',
                  '2026-08-21T00:00:00+00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO model_selections (
            selection_id, target, promotion_id, supersedes_selection_id, selected_at_utc
        ) VALUES (1, 'inunda', 1, NULL, '2026-08-21T00:00:00+00:00')
        """
    )
    conn.commit()

    shutil.copyfile(
        SQL_DIR / "005_provenance_integrity.sql",
        catalog / "005_provenance_integrity.sql",
    )
    apply_migrations(conn, migration_dir=catalog)

    invalidated = conn.execute(
        "SELECT reason FROM model_promotion_invalidations WHERE promotion_id = 1"
    ).fetchone()
    assert invalidated is not None

    active = conn.execute("SELECT * FROM active_model_selections").fetchall()
    assert active == []  # upgraded DB may have no active selection for any target

    valid = conn.execute(
        "SELECT 1 FROM valid_model_promotions WHERE promotion_id = 1"
    ).fetchall()
    assert valid == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_migration_005_legacy_promotions.py -v`
Expected: FAIL — `sqlite3.OperationalError: no such table: model_promotion_invalidations`

- [ ] **Step 3: Append Part F, closing out the migration**

```sql
CREATE TABLE model_promotion_invalidations (
    promotion_id INTEGER PRIMARY KEY
        REFERENCES model_promotions(promotion_id) ON DELETE RESTRICT,
    reason TEXT NOT NULL,
    invalidated_at_utc TEXT NOT NULL
);

CREATE TRIGGER model_promotion_invalidations_identity_conflict
BEFORE INSERT ON model_promotion_invalidations
WHEN EXISTS (
    SELECT 1 FROM model_promotion_invalidations WHERE promotion_id=NEW.promotion_id
)
BEGIN
    SELECT RAISE(ABORT, 'model promotion invalidation identity is immutable');
END;

CREATE TRIGGER model_promotion_invalidations_no_update
BEFORE UPDATE ON model_promotion_invalidations
BEGIN
    SELECT RAISE(ABORT, 'model promotion invalidations are immutable');
END;

CREATE TRIGGER model_promotion_invalidations_no_delete
BEFORE DELETE ON model_promotion_invalidations
BEGIN
    SELECT RAISE(ABORT, 'model promotion invalidations are immutable');
END;

CREATE VIEW valid_model_promotions AS
SELECT model_promotions.*
FROM model_promotions
JOIN model_promotion_finalizations
    ON model_promotion_finalizations.promotion_id = model_promotions.promotion_id
LEFT JOIN model_promotion_invalidations AS invalidation
    ON invalidation.promotion_id = model_promotions.promotion_id
WHERE invalidation.promotion_id IS NULL;

-- Rebuild active_model_selections: determine the unique leaf over the
-- complete immutable selection chain FIRST, then join validity. Filtering
-- promotions before leaf selection would silently reactivate an older model.
DROP VIEW active_model_selections;

CREATE VIEW active_model_selections AS
WITH leaf_selections AS (
    SELECT selection.*
    FROM model_selections AS selection
    WHERE NOT EXISTS (
        SELECT 1 FROM model_selections AS successor
        WHERE successor.supersedes_selection_id = selection.selection_id
    )
)
SELECT
    leaf_selections.selection_id,
    leaf_selections.target,
    leaf_selections.promotion_id,
    leaf_selections.selected_at_utc,
    valid_model_promotions.training_run_id,
    valid_model_promotions.classifier_model_id,
    valid_model_promotions.regressor_model_id,
    valid_model_promotions.primary_metric,
    valid_model_promotions.primary_value,
    valid_model_promotions.tie_breakers_json,
    valid_model_promotions.ranking_json,
    valid_model_promotions.promoted_at_utc
FROM leaf_selections
JOIN valid_model_promotions
    ON valid_model_promotions.promotion_id = leaf_selections.promotion_id
   AND valid_model_promotions.target = leaf_selections.target;

-- Invalidate every promotion that existed before 005: none can prove a
-- complete frozen ranking universe under the new schema.
INSERT INTO model_promotion_invalidations (promotion_id, reason, invalidated_at_utc)
SELECT
    promotion_id,
    CASE
        WHEN ranking_json LIKE '%"source": "001_v17_initial"%'
             AND target = 'system'
        THEN 'fabricated_system_value'
        WHEN ranking_json LIKE '%"source": "001_v17_initial"%'
        THEN 'missing_target_evidence'
        ELSE 'missing_normalized_ranking_evidence'
    END,
    datetime('now')
FROM model_promotions;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_migration_005_legacy_promotions.py -v`
Expected: PASS

- [ ] **Step 5: Run the entire migration 005 test set together against the packaged catalog**

```python
# tests/database/test_migration_005_full_catalog.py
from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations


def test_005_applies_cleanly_from_a_fresh_database(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)  # packaged catalog: 001..005
    applied = [row[0] for row in conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()]
    assert applied == [1, 2, 3, 4, 5]


def test_005_is_idempotent(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    apply_migrations(conn)  # second call must be a no-op, not an error
    applied = [row[0] for row in conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()]
    assert applied == [1, 2, 3, 4, 5]
```

Run: `python -m pytest tests/database/test_migration_005_full_catalog.py -v`
Expected: PASS (2 tests) — proves 005 is now part of the packaged catalog and self-consistent end to end

- [ ] **Step 6: Run the complete pre-existing database test suite**

Run: `python -m pytest tests/database -v`
Expected: PASS — every test written for 001-004 in Tasks 4-9's fixtures still passes; `test_migrations_v17.py`'s existing 001-004-only tests are unaffected since they build explicit subset catalogs

- [ ] **Step 7: Commit**

```bash
git add swmm_resilience/database/sql/005_provenance_integrity.sql tests/database/test_migration_005_legacy_promotions.py tests/database/test_migration_005_full_catalog.py
git commit -m "feat: migration 005 part F - invalidate legacy promotions, rebuild active-selection view"
```

---

## Task 10: Wire the 005 Python preflight validator

**Files:**
- Create: `swmm_resilience/database/migration_005_validator.py`
- Modify: `swmm_resilience/database/migrations.py`
- Test: `tests/database/test_migration_005_preflight.py`

**Interfaces:**
- `def validate_before_005(conn: sqlite3.Connection) -> None` — the versioned preflight hook. Runs the read-only checks spec section 4 requires before any 005 DDL: no non-prefix migration history / unexpected checksum (already guaranteed by `_validate_applied_history`, re-asserted here defensively), no existing `PRAGMA foreign_key_check`/`PRAGMA integrity_check` violation, and that every pre-005 training run and promotion can be enumerated (a plain `SELECT training_run_id FROM training_runs` / `SELECT promotion_id FROM model_promotions` that must not raise). Raises `MigrationPreflightError` (from Task 1) with a typed, no-BLOB-contents message on any failure.
- `DEFAULT_PREFLIGHT_HOOKS: dict[int, Callable]` = `{5: validate_before_005}` — the production hook table.
- `apply_migrations`'s default when `preflight_hooks=None` becomes `DEFAULT_PREFLIGHT_HOOKS`, not `{}` — so callers get 005's preflight automatically without having to know about it (Task 1 left the default as `{}`; this task changes that default and records the validator's own checksum).

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_005_preflight.py
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import (
    MigrationPreflightError,
    apply_migrations,
)


def test_005_preflight_runs_by_default_and_records_validator_checksum(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    row = conn.execute(
        "SELECT validator_name, validator_sha256 FROM schema_migration_validators WHERE version = 5"
    ).fetchone()
    assert row is not None
    assert row[0] == "validate_before_005"
    assert len(row[1]) == 64


def test_005_preflight_aborts_on_existing_foreign_key_violation(tmp_path, monkeypatch):
    conn = connect_database(tmp_path / "db.sqlite3")

    from swmm_resilience.database import migration_005_validator

    def fake_fk_check(_conn):
        return [("training_runs", 1, "training_runs", 0)]  # simulate a violation

    monkeypatch.setattr(
        migration_005_validator, "_foreign_key_violations", fake_fk_check
    )
    with pytest.raises(MigrationPreflightError):
        apply_migrations(conn)
    applied = [row[0] for row in conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()]
    assert applied == [1, 2, 3, 4]  # 005 did not commit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_migration_005_preflight.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swmm_resilience.database.migration_005_validator'`

- [ ] **Step 3: Implement the validator and wire the default**

```python
# swmm_resilience/database/migration_005_validator.py
from __future__ import annotations

import sqlite3

from .migrations import MigrationPreflightError


def _foreign_key_violations(conn: sqlite3.Connection):
    return conn.execute("PRAGMA foreign_key_check").fetchall()


def _integrity_violations(conn: sqlite3.Connection):
    rows = conn.execute("PRAGMA integrity_check").fetchall()
    return [] if rows == [("ok",)] else rows


def validate_before_005(conn: sqlite3.Connection) -> None:
    fk_violations = _foreign_key_violations(conn)
    if fk_violations:
        raise MigrationPreflightError(
            f"Migration 005 preflight: {len(fk_violations)} foreign-key "
            "violation(s) exist before upgrade; refusing to proceed"
        )

    integrity_violations = _integrity_violations(conn)
    if integrity_violations:
        raise MigrationPreflightError(
            f"Migration 005 preflight: {len(integrity_violations)} "
            "integrity_check violation(s) exist before upgrade; refusing to proceed"
        )

    try:
        conn.execute("SELECT training_run_id FROM training_runs").fetchall()
        conn.execute("SELECT promotion_id FROM model_promotions").fetchall()
    except sqlite3.DatabaseError as exc:
        raise MigrationPreflightError(
            "Migration 005 preflight: cannot enumerate pre-005 training runs "
            "and promotions for deterministic invalidation"
        ) from exc
```

In `swmm_resilience/database/migrations.py`, add near the bottom (after `apply_migrations` is defined, to avoid a circular import at module load time — the validator module imports `MigrationPreflightError` from here):

```python
def _default_preflight_hooks():
    from . import migration_005_validator

    return {5: migration_005_validator.validate_before_005}
```

Change the `apply_migrations` signature default and add checksum recording:

```python
def apply_migrations(
    conn: sqlite3.Connection,
    migration_dir: Path | None = None,
    preflight_hooks: dict[int, "Callable[[sqlite3.Connection], None]"] | None = None,
) -> None:
    if conn.in_transaction:
        raise RuntimeError(
            "Cannot apply migrations while the connection has an active transaction"
        )

    hooks = preflight_hooks if preflight_hooks is not None else _default_preflight_hooks()
    catalog = _migration_catalog(migration_dir)
    applied = _applied_history(conn)
    _validate_applied_history(applied, catalog)

    applied_any = False
    for version, name, sql, checksum in catalog[len(applied):]:
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute("BEGIN IMMEDIATE")
            hook = hooks.get(version)
            if hook is not None:
                hook(conn)
            _execute_migration_sql(conn, sql)
            conn.execute(
                """
                INSERT INTO schema_migrations (
                    version, name, checksum_sha256, applied_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (version, name, checksum, stamp),
            )
            if hook is not None:
                validator_source = inspect.getsource(hook)
                validator_checksum = hashlib.sha256(
                    validator_source.encode("utf-8")
                ).hexdigest()
                conn.execute(
                    """
                    INSERT INTO schema_migration_validators (
                        version, validator_name, validator_sha256
                    ) VALUES (?, ?, ?)
                    """,
                    (version, hook.__name__, validator_checksum),
                )
            conn.commit()
            applied_any = True
        except Exception:
            conn.rollback()
            raise

    if applied_any:
        conn.execute("PRAGMA optimize")
```

Add `import inspect` to the top-level imports in `migrations.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_migration_005_preflight.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Re-run Task 1 and Task 4-9 tests to confirm the new default hook doesn't break anything**

Run: `python -m pytest tests/database/test_migration_preflight_hooks.py tests/database -v`
Expected: PASS — Task 1's tests pass explicit `preflight_hooks={...}` so they're unaffected by the new default; every 005 test applies the real packaged 005 SQL and now also gets its preflight validator recorded

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/migration_005_validator.py swmm_resilience/database/migrations.py tests/database/test_migration_005_preflight.py
git commit -m "feat: wire migration 005 preflight validator with recorded checksum"
```

**Deferred, not dropped:** spec section 4 also requires the recorded validator checksum to be re-verified "on every later migration run," not just recorded once when its migration applies. There is no migration 006 yet, so there's nothing to re-verify against today. When a future migration is added, `apply_migrations` must, before running that migration's own hook, recompute every already-recorded `schema_migration_validators` row's checksum from the currently-importable hook function and abort if any differ from what's stored. Add that check (and its test — tamper with a hook function via `monkeypatch` after 005 is applied, then attempt to apply a hypothetical migration 006, and assert `MigrationPreflightError`) at that time, not speculatively now (YAGNI — there's no second hook to verify against yet).

---

## Task 11: Real-only feature contract — reject complex dtypes

**Files:**
- Modify: `swmm_resilience/ml/contracts.py`
- Test: `tests/ml/test_feature_contract_v17.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/ml/test_feature_contract_v17.py
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.contracts import TABULAR_V3_17, FeatureContractError


def _valid_frame():
    return pd.DataFrame(
        {name: [1.0, 2.0] for name in TABULAR_V3_17.feature_names}
    )


def test_rejects_complex_dtype_even_with_zero_imaginary_component():
    frame = _valid_frame()
    frame["elev_fondo"] = pd.array(
        [1 + 0j, 2 + 0j], dtype=complex
    )
    with pytest.raises(FeatureContractError, match="complex"):
        TABULAR_V3_17.validate_frame(frame)


def test_rejects_complex_dtype_with_nonzero_imaginary_component():
    frame = _valid_frame()
    frame["elev_fondo"] = pd.array([1 + 1j, 2 + 0j], dtype=complex)
    with pytest.raises(FeatureContractError, match="complex"):
        TABULAR_V3_17.validate_frame(frame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ml/test_feature_contract_v17.py -v -k complex`
Expected: FAIL — no exception raised (numpy silently truncates today)

- [ ] **Step 3: Add the explicit dtype gate**

In `swmm_resilience/ml/contracts.py`, add near the top-level imports:

```python
from pandas.api.types import is_complex_dtype
```

In `FeatureContract.validate_frame`, insert the complex-dtype check before the existing `bad_types` loop (which currently only tests `is_numeric_dtype`/`is_bool_dtype` and would let complex columns through into the failing `astype(float)` truncation):

```python
    def validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        actual = tuple(frame.columns)
        if actual != self.feature_names:
            raise FeatureContractError(
                f"Expected ordered features {self.feature_names}; received {actual}"
            )
        if frame.empty:
            raise FeatureContractError("Feature frame is empty")
        complex_columns = [
            name for name in self.feature_names if is_complex_dtype(frame[name])
        ]
        if complex_columns:
            raise FeatureContractError(
                f"Complex-valued feature columns are not permitted: {complex_columns}"
            )
        bad_types = []
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ml/test_feature_contract_v17.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/contracts.py tests/ml/test_feature_contract_v17.py
git commit -m "fix: reject complex-valued feature columns instead of silently truncating"
```

---

## Task 12: Git ignore rules for SQLite sidecar files

**Files:**
- Modify: `.gitignore`
- Test: `tests/database/test_gitignore_sqlite_sidecars.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_gitignore_sqlite_sidecars.py
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]

SIDECAR_NAMES = [
    "outputs/models.sqlite3",
    "outputs/models.sqlite3-wal",
    "outputs/models.sqlite3-shm",
    "outputs/models.sqlite3-journal",
    "outputs/models.sqlite",
    "outputs/models.sqlite-journal",
    "outputs/models.db-wal",
    "outputs/models.db-shm",
    "outputs/models.workflow.lock",
]


@pytest.mark.parametrize("relative_path", SIDECAR_NAMES)
def test_sqlite_sidecar_paths_are_git_ignored(relative_path, tmp_path):
    target = REPO_ROOT / relative_path
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(target)],
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"{relative_path} is not git-ignored"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_gitignore_sqlite_sidecars.py -v`
Expected: FAIL for most parametrized paths (only the pre-existing `*.db` rule matches today)

- [ ] **Step 3: Add the ignore rules**

Append to `.gitignore`:

```
*.sqlite3
*.sqlite3-wal
*.sqlite3-shm
*.sqlite3-journal
*.sqlite
*.sqlite-journal
*.db-wal
*.db-shm
*.db-journal
*.workflow.lock
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_gitignore_sqlite_sidecars.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add .gitignore tests/database/test_gitignore_sqlite_sidecars.py
git commit -m "build: ignore SQLite database and sidecar files"
```

---

## Task 13: `upgrade_database_with_backup()`

**Files:**
- Create: `swmm_resilience/database/upgrade.py`
- Test: `tests/database/test_upgrade_v17.py`

**Interfaces:**
- `class UpgradeReceipt` (frozen dataclass): `source_path`, `backup_path`, `backup_sha256`, `schema_version_before`, `logical_fingerprint`.
- `def upgrade_database_with_backup(database_path: str | Path, backup_dir: str | Path) -> UpgradeReceipt` — acquires `WorkflowLock(database_path)`; opens a managed connection; if already at the latest migration version, returns early without a backup; otherwise checkpoints WAL, creates a timestamped backup via `checkpoint_and_backup`, verifies the backup's `PRAGMA integrity_check`, computes a logical fingerprint (`PRAGMA schema_version` combined with a content hash — see Step 3), builds a single-use `UpgradeReceipt`, re-validates the fingerprint hasn't changed since the backup was taken (guards against a concurrent unmanaged WAL writer), then calls `apply_migrations(conn)` while still holding the lock.

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_upgrade_v17.py
from pathlib import Path

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.upgrade import upgrade_database_with_backup
from swmm_resilience.database.workflow_lock import WorkflowLock, WorkflowLockError


def _make_v4_database(path):
    conn = connect_database(path)
    # Simulate a database stuck at version 4 by monkeypatching the catalog
    # is unnecessary here: applying the full packaged catalog then manually
    # deleting the 005 row + its tables would violate FKs, so instead we
    # apply through an explicit 001-004-only directory.
    conn.close()


def test_upgrade_backs_up_before_applying_005(tmp_path):
    from swmm_resilience.database.migrations import _migration_catalog
    import shutil

    sql_dir = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql",
        "002_model_integrity.sql",
        "003_model_integrity_guards.sql",
        "004_training_run_identity.sql",
    ):
        shutil.copyfile(sql_dir / name, catalog / name)

    db_path = tmp_path / "db.sqlite3"
    conn = connect_database(db_path)
    apply_migrations(conn, migration_dir=catalog)
    conn.close()

    backup_dir = tmp_path / "backups"
    receipt = upgrade_database_with_backup(db_path, backup_dir)

    assert receipt.backup_path.exists()
    assert receipt.schema_version_before == 4

    verify = connect_database(db_path)
    applied = [row[0] for row in verify.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()]
    assert applied == [1, 2, 3, 4, 5]
    verify.close()


def test_upgrade_is_a_noop_when_already_current(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    conn = connect_database(db_path)
    apply_migrations(conn)  # already at 005
    conn.close()

    receipt = upgrade_database_with_backup(db_path, tmp_path / "backups")
    assert receipt.backup_path is None


def test_upgrade_fails_when_workflow_lock_already_held(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    conn = connect_database(db_path)
    apply_migrations(conn)
    conn.close()

    with WorkflowLock(db_path):
        with pytest.raises(WorkflowLockError):
            upgrade_database_with_backup(db_path, tmp_path / "backups")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_upgrade_v17.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swmm_resilience.database.upgrade'`

- [ ] **Step 3: Implement**

```python
# swmm_resilience/database/upgrade.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3

from .connection import connect_database
from .maintenance import checkpoint_and_backup
from .migrations import apply_migrations
from .workflow_lock import WorkflowLock


class UpgradeIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class UpgradeReceipt:
    source_path: Path
    backup_path: Path | None
    backup_sha256: str | None
    schema_version_before: int
    logical_fingerprint: str | None


def _current_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return row[0] or 0


def _latest_catalog_version() -> int:
    from .migrations import _migration_catalog

    return max(version for version, _name, _sql, _checksum in _migration_catalog(None))


def _logical_fingerprint(conn: sqlite3.Connection) -> str:
    schema_version = conn.execute("PRAGMA schema_version").fetchone()[0]
    checksums = conn.execute(
        "SELECT checksum_sha256 FROM schema_migrations ORDER BY version"
    ).fetchall()
    payload = f"{schema_version}:{[row[0] for row in checksums]}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upgrade_database_with_backup(
    database_path: str | Path,
    backup_dir: str | Path,
) -> UpgradeReceipt:
    source_path = Path(database_path)
    with WorkflowLock(source_path):
        conn = connect_database(source_path)
        try:
            schema_version_before = _current_schema_version(conn)
            if schema_version_before >= _latest_catalog_version():
                return UpgradeReceipt(
                    source_path=source_path,
                    backup_path=None,
                    backup_sha256=None,
                    schema_version_before=schema_version_before,
                    logical_fingerprint=None,
                )

            fingerprint_before = _logical_fingerprint(conn)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = Path(backup_dir) / f"{source_path.stem}.v{schema_version_before}.{stamp}.sqlite3"
            checkpoint_and_backup(conn, backup_path)

            backup_conn = sqlite3.connect(backup_path)
            try:
                integrity = backup_conn.execute("PRAGMA integrity_check").fetchall()
                if integrity != [("ok",)]:
                    raise UpgradeIntegrityError(
                        f"Backup integrity check failed: {integrity}"
                    )
            finally:
                backup_conn.close()
            backup_sha256 = _file_sha256(backup_path)

            fingerprint_after = _logical_fingerprint(conn)
            if fingerprint_after != fingerprint_before:
                raise UpgradeIntegrityError(
                    "Database changed between backup and migration; refusing to proceed"
                )

            apply_migrations(conn)

            return UpgradeReceipt(
                source_path=source_path,
                backup_path=backup_path,
                backup_sha256=backup_sha256,
                schema_version_before=schema_version_before,
                logical_fingerprint=fingerprint_after,
            )
        finally:
            conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_upgrade_v17.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/database/upgrade.py tests/database/test_upgrade_v17.py
git commit -m "feat: add upgrade_database_with_backup with locked, verified 004-to-005 upgrade"
```

---

## Task 14: Interrupted-run recovery operation

**Files:**
- Create: `swmm_resilience/database/recovery.py`
- Test: `tests/database/test_recovery_v17.py`

**Interfaces:**
- `def recover_abandoned_training_run(conn: sqlite3.Connection, training_run_id: int) -> None` — acquires the caller-supplied connection's implicit transaction (`BEGIN IMMEDIATE`); requires the target training run's status to be `RUNNING` (raises `ValueError` otherwise — matches spec 4.1's "a promoted or terminal run cannot be recovered this way"); transitions every `PENDING`/`RUNNING` evaluation under that run to `FAILED` with a fixed reason and timestamp, then transitions the training run itself to `FAILED`. Does not delete any row. This function assumes its caller already holds the process-level `WorkflowLock` — it does not acquire one itself, so it can be composed by a caller (CLI/UI) that also needs the lock for other setup.

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_recovery_v17.py
import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.recovery import recover_abandoned_training_run


def _running_training_run_with_evaluations(conn):
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'RUNNING')
        """,
        ("a" * 64,),
    )
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (1, 1, 'classification', 'xgboost', '{}', 0, '[]', '[]', 'PENDING', 0, 0)
        """
    )
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (2, 1, 'classification', 'xgboost', '{}', 1, '[]', '[]', 'COMPLETE', 1, 1)
        """
    )
    conn.commit()


def test_recovery_marks_running_run_and_pending_evaluations_failed(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    _running_training_run_with_evaluations(conn)

    recover_abandoned_training_run(conn, training_run_id=1)

    run_status = conn.execute(
        "SELECT status, failure_type FROM training_runs WHERE training_run_id = 1"
    ).fetchone()
    assert run_status[0] == "FAILED"
    assert run_status[1] == "interrupted_run_recovery"

    eval_statuses = dict(conn.execute(
        "SELECT evaluation_id, status FROM model_evaluations WHERE training_run_id = 1"
    ).fetchall())
    assert eval_statuses[1] == "FAILED"   # was PENDING -> now FAILED
    assert eval_statuses[2] == "COMPLETE"  # already-complete evidence is untouched


def test_recovery_rejects_non_running_training_run(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'PENDING')
        """,
        ("a" * 64,),
    )
    conn.commit()

    with pytest.raises(ValueError):
        recover_abandoned_training_run(conn, training_run_id=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_recovery_v17.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swmm_resilience.database.recovery'`

- [ ] **Step 3: Implement**

```python
# swmm_resilience/database/recovery.py
from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

RECOVERY_REASON = "interrupted_run_recovery"


def recover_abandoned_training_run(conn: sqlite3.Connection, training_run_id: int) -> None:
    row = conn.execute(
        "SELECT status FROM training_runs WHERE training_run_id = ?",
        (training_run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Training run {training_run_id} does not exist")
    if row[0] != "RUNNING":
        raise ValueError(
            f"Training run {training_run_id} is {row[0]!r}; only a RUNNING "
            "run can be recovered"
        )

    stamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE model_evaluations
        SET status = 'FAILED', failure_type = ?, failure_message = ?
        WHERE training_run_id = ? AND status IN ('PENDING', 'RUNNING')
        """,
        (RECOVERY_REASON, "Marked failed by interrupted-run recovery", training_run_id),
    )
    conn.execute(
        """
        UPDATE training_runs
        SET status = 'FAILED', failure_type = ?, failure_message = ?
        WHERE training_run_id = ?
        """,
        (RECOVERY_REASON, "Marked failed by interrupted-run recovery", training_run_id),
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_recovery_v17.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Add the "cannot recover a promoted run" regression**

```python
# append to tests/database/test_recovery_v17.py
def test_recovery_rejects_promoted_training_run(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    _running_training_run_with_evaluations(conn)
    conn.execute(
        "UPDATE model_evaluations SET status = 'COMPLETE' WHERE evaluation_id = 1"
    )
    conn.execute(
        "UPDATE training_runs SET status = 'COMPLETE' WHERE training_run_id = 1"
    )
    conn.commit()

    with pytest.raises(ValueError):
        recover_abandoned_training_run(conn, training_run_id=1)
```

Run: `python -m pytest tests/database/test_recovery_v17.py -v`
Expected: PASS (3 tests) — `COMPLETE` is already rejected by the existing status check, no implementation change needed

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/recovery.py tests/database/test_recovery_v17.py
git commit -m "feat: add explicit interrupted-run recovery operation"
```

---

## Task 15: Pin source simulation rows and allowlist the training query

**Files:**
- Modify: `swmm_resilience/database/sql/005_provenance_integrity.sql` (append — insert this before Part F's legacy-invalidation data migration, since it adds columns/triggers on tables Part F's view rebuild doesn't touch, but must exist before any new operational training could occur)
- Test: `tests/database/test_migration_005_pinning.py`

Spec 3.2.1: inserting a run into `training_run_inputs` must pin the exact tabular sample source rows, and operational training must use only the allowlisted canonical query (`training_samples_v17`), not arbitrary `query_sql` text.

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_005_pinning.py
from pathlib import Path
import shutil

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def _seed_pinnable_run(conn):
    conn.execute(
        """
        INSERT INTO networks (network_id, network_sha256, name, source_filename,
                               inp_bytes, flow_units, created_at_utc)
        VALUES (1, ?, 'net', 'net.inp', ?, 'LPS', '2026-08-22T00:00:00+00:00')
        """,
        ("f" * 64, b"inp-bytes"),
    )
    conn.execute(
        "INSERT INTO nodes (node_pk, network_id, node_uid) VALUES (1, 1, 'N1')"
    )
    conn.execute(
        "INSERT INTO runs (run_id, network_id, scenario_id, status, node_count) "
        "VALUES (1, 1, NULL, 'COMPLETE', 1)"
    )
    conn.execute(
        "INSERT INTO node_results (run_id, node_pk, inunda, vol_inundacion_m3) "
        "VALUES (1, 1, 0, 0.0)"
    )
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status, training_query_contract_id,
            training_query_contract_sha256
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[1]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'PENDING',
                  'training_samples_v17', ?)
        """,
        ("a" * 64, "d" * 64),
    )
    conn.commit()


def test_pinned_node_results_row_cannot_be_updated_or_deleted(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql", "002_model_integrity.sql",
        "003_model_integrity_guards.sql", "004_training_run_identity.sql",
        "005_provenance_integrity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_pinnable_run(conn)
    conn.execute("INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)")
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("UPDATE node_results SET inunda = 1 WHERE run_id = 1 AND node_pk = 1")
    conn.rollback()
    with pytest.raises(Exception):
        conn.execute("DELETE FROM node_results WHERE run_id = 1 AND node_pk = 1")
    conn.rollback()
    with pytest.raises(Exception):
        conn.execute("DELETE FROM runs WHERE run_id = 1")
    conn.rollback()


def test_training_run_requires_allowlisted_query_contract(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql", "002_model_integrity.sql",
        "003_model_integrity_guards.sql", "004_training_run_identity.sql",
        "005_provenance_integrity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO training_runs (
                training_run_id, target, feature_contract_id,
                feature_contract_sha256, query_sql, query_params_json,
                included_run_ids_json, grouping_strategy, fold_count, random_seed,
                primary_metric, tie_breakers_json, python_version,
                library_versions_json, status, training_query_contract_id,
                training_query_contract_sha256
            ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                      'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'PENDING',
                      'arbitrary_query', ?)
            """,
            ("a" * 64, "d" * 64),
        )
    conn.rollback()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/database/test_migration_005_pinning.py -v`
Expected: FAIL — `sqlite3.OperationalError: table training_runs has no column named training_query_contract_id`

- [ ] **Step 3: Append the pinning columns and triggers**

```sql
ALTER TABLE training_runs
    ADD COLUMN training_query_contract_id TEXT NOT NULL DEFAULT 'training_samples_v17'
        CHECK(training_query_contract_id='training_samples_v17');
ALTER TABLE training_runs
    ADD COLUMN training_query_contract_sha256 TEXT NOT NULL DEFAULT ''
        CHECK(length(training_query_contract_sha256)=64);

-- Pin every row training_run_inputs references: the run itself, its scenario
-- and network, its nodes, and its node_features/node_results. Cleanup must
-- report "pinned by training provenance" instead of silently deleting.
CREATE TRIGGER runs_pinned_no_update
BEFORE UPDATE ON runs
WHEN EXISTS (
    SELECT 1 FROM training_run_inputs WHERE run_id=OLD.run_id
) AND (NEW.status IS NOT OLD.status OR NEW.network_id IS NOT OLD.network_id)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER runs_pinned_no_delete
BEFORE DELETE ON runs
WHEN EXISTS (SELECT 1 FROM training_run_inputs WHERE run_id=OLD.run_id)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER node_features_pinned_no_write
BEFORE UPDATE ON node_features
WHEN EXISTS (SELECT 1 FROM training_run_inputs WHERE run_id=OLD.run_id)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER node_features_pinned_no_delete
BEFORE DELETE ON node_features
WHEN EXISTS (SELECT 1 FROM training_run_inputs WHERE run_id=OLD.run_id)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER node_features_pinned_no_insert
BEFORE INSERT ON node_features
WHEN EXISTS (SELECT 1 FROM training_run_inputs WHERE run_id=NEW.run_id)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER node_results_pinned_no_write
BEFORE UPDATE ON node_results
WHEN EXISTS (SELECT 1 FROM training_run_inputs WHERE run_id=OLD.run_id)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER node_results_pinned_no_delete
BEFORE DELETE ON node_results
WHEN EXISTS (SELECT 1 FROM training_run_inputs WHERE run_id=OLD.run_id)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER node_results_pinned_no_insert
BEFORE INSERT ON node_results
WHEN EXISTS (SELECT 1 FROM training_run_inputs WHERE run_id=NEW.run_id)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER nodes_pinned_no_write
BEFORE UPDATE ON nodes
WHEN EXISTS (
    SELECT 1 FROM training_run_inputs
    JOIN node_features ON node_features.run_id = training_run_inputs.run_id
    WHERE node_features.node_pk = OLD.node_pk
)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER nodes_pinned_no_delete
BEFORE DELETE ON nodes
WHEN EXISTS (
    SELECT 1 FROM training_run_inputs
    JOIN node_features ON node_features.run_id = training_run_inputs.run_id
    WHERE node_features.node_pk = OLD.node_pk
)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER networks_pinned_no_write
BEFORE UPDATE ON networks
WHEN EXISTS (
    SELECT 1 FROM training_run_inputs
    JOIN runs ON runs.run_id = training_run_inputs.run_id
    WHERE runs.network_id = OLD.network_id
)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER networks_pinned_no_delete
BEFORE DELETE ON networks
WHEN EXISTS (
    SELECT 1 FROM training_run_inputs
    JOIN runs ON runs.run_id = training_run_inputs.run_id
    WHERE runs.network_id = OLD.network_id
)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

-- training_run_inputs insertion requires the network's SHA-256 to match its
-- stored bytes (managed connections only: sha256() must be registered).
CREATE TRIGGER training_run_inputs_verifies_network_hash
BEFORE INSERT ON training_run_inputs
WHEN NOT EXISTS (
    SELECT 1
    FROM runs
    JOIN networks ON networks.network_id = runs.network_id
    WHERE runs.run_id = NEW.run_id
      AND runs.status = 'COMPLETE'
      AND networks.network_sha256 = sha256(networks.inp_bytes)
)
BEGIN
    SELECT RAISE(ABORT, 'pinned run must be COMPLETE with a verified network hash');
END;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/database/test_migration_005_pinning.py -v`
Expected: PASS. Because `training_run_inputs_verifies_network_hash` calls `sha256()`, every test in Tasks 4-9 and this task that inserts into `training_run_inputs` must now run on a connection from `connect_managed_database` (Task 3), not `connect_database` — go back and change the `connect_database` import/calls to `connect_managed_database` in `tests/database/test_migration_005_membership.py`, `test_migration_005_oof.py`, and this file's own tests; a raw connection now fails closed on any such insert, which is the intended behavior, not a bug to work around.

- [ ] **Step 5: Re-run every migration-005 test with the corrected imports**

Run: `python -m pytest tests/database -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/database/sql/005_provenance_integrity.sql tests/database/test_migration_005_pinning.py tests/database/test_migration_005_membership.py tests/database/test_migration_005_oof.py
git commit -m "feat: migration 005 - pin source simulation rows and allowlist the training query contract"
```

---

## Task 16: Query-plan and scale gates

**Files:**
- Modify: `tests/database/test_query_plans_v17.py`
- Modify: `tests/database/test_scale_v17.py`

- [ ] **Step 1: Add an EXPLAIN QUERY PLAN gate for `active_model_selections`**

Read `tests/database/test_query_plans_v17.py` first to match its existing assertion style (it already gates `training_samples_v17`). Append a test in the same style:

```python
def test_active_model_selections_query_plan_has_no_full_table_scan(tmp_path):
    from swmm_resilience.database.connection import connect_database
    from swmm_resilience.database.migrations import apply_migrations

    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    plan_rows = conn.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM active_model_selections"
    ).fetchall()
    full_scans = [
        row for row in plan_rows
        if "SCAN" in row[3] and "USING INDEX" not in row[3] and "SCAN TABLE model_selections" not in row[3]
    ]
    # model_selections itself is expected to be scanned (it's small and has
    # no covering index for "no successor" — the join targets must use one).
    assert not any("SCAN oof_predictions" in row[3] for row in plan_rows)
    assert not any("SCAN model_promotions" in row[3] and "USING INDEX" not in row[3] for row in plan_rows)
```

- [ ] **Step 2: Run and adjust**

Run: `python -m pytest tests/database/test_query_plans_v17.py -v`
Expected: PASS. If it fails because `model_promotions` is table-scanned, add `CREATE INDEX idx_promotions_lookup ON model_promotions(promotion_id, target)` to migration 005 (append to Part F) and re-run — indexes are additive and safe to add after Task 9's commit, in a follow-up small commit.

- [ ] **Step 3: Add an opt-in one-million-row OOF scale fixture**

Read `tests/database/test_scale_v17.py` first to reuse its existing scale-fixture helpers (network/run/node generation) and its `@pytest.mark.scale` marker convention. Add:

```python
@pytest.mark.scale
def test_one_million_oof_rows_finalize_and_rerank_within_bounds(tmp_path):
    import time

    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    # Reuse this file's existing scale-fixture builder to create one
    # training run, one candidate, one evaluation per fold, and enough
    # runs/nodes that their OOF rows total >= 1,000,000, following the
    # same fixture-construction helper already used by the node_results
    # scale test above in this file.
    started = time.monotonic()
    # ... insert candidate, finalize it (see Task 8's fixtures for the
    # exact insert shapes) ...
    elapsed = time.monotonic() - started
    assert elapsed < 120
    db_size_mib = (tmp_path / "db.sqlite3").stat().st_size / (1024 * 1024)
    assert db_size_mib < 500
```

Mark this test to be filled in with the concrete row-generation loop by the engineer executing this task, using the exact fixture helpers already present in `test_scale_v17.py` (do not duplicate the 1M-row generator — call it). Run the file's existing scale tests first to confirm the marker/CI-skip convention:

Run: `python -m pytest tests/database/test_scale_v17.py -v -m "not scale"`
Expected: PASS, scale test deselected by default (matches this repo's existing 2-deselected pattern seen in the Task 4 baseline run)

- [ ] **Step 4: Commit**

```bash
git add tests/database/test_query_plans_v17.py tests/database/test_scale_v17.py swmm_resilience/database/sql/005_provenance_integrity.sql
git commit -m "test: add active_model_selections query-plan gate and OOF scale fixture"
```

---

## Task 17: Packaging verification and full-suite sign-off

**Files:**
- Create: `tests/packaging/test_wheel_migrations_v17.py` (new directory)
- No other modifications — this task only verifies everything already built

- [ ] **Step 1: Write the packaging test**

```python
# tests/packaging/test_wheel_migrations_v17.py
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_wheel_exposes_all_five_migrations_and_validator(tmp_path):
    dist_dir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    wheel = next(dist_dir.glob("*.whl"))

    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", str(wheel)], check=True)

    script = (
        "from importlib.resources import files\n"
        "root = files('swmm_resilience.database').joinpath('sql')\n"
        "names = sorted(p.name for p in root.iterdir() if p.name.endswith('.sql'))\n"
        "assert names == ["
        "'001_v17_initial.sql','002_model_integrity.sql',"
        "'003_model_integrity_guards.sql','004_training_run_identity.sql',"
        "'005_provenance_integrity.sql'], names\n"
        "from swmm_resilience.database import migration_005_validator\n"
        "assert hasattr(migration_005_validator, 'validate_before_005')\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [str(python), "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/packaging/test_wheel_migrations_v17.py -v`
Expected: PASS. If `build` is not installed in the venv, run `python -m pip install build` first — this is a dev-only tool, not a runtime dependency, so it does not go in `requirements.txt`.

- [ ] **Step 3: Run the complete test suite**

Run: `python -m pytest -v`
Expected: PASS, with the pre-existing 104-passed/2-deselected database baseline now grown by every test added in Tasks 1-15, plus this task's packaging test, plus the deselected `@pytest.mark.scale` test from Task 15.

- [ ] **Step 4: Cross-check against spec section 6's 20-item verification list**

Open `docs/superpowers/specs/2026-08-22-sqlite-v17-provenance-hardening-design.md` section 6 side by side with the tests written in Tasks 4-15. Confirm each of the 20 items has at least one covering test:

| Spec item | Covered by |
|---|---|
| 1. rowid/_rowid_/oid identity updates | Task 4 |
| 2. permitted/forbidden state transitions | Task 5 |
| 3. mutation/deletion/UPSERT/REPLACE rejection | Tasks 5-9 (per-table no-update/no-delete/identity-conflict triggers) |
| 4. OOF domain/membership checks | Task 7 |
| 5. candidate/artifact mismatches | Task 8 |
| 6. promotion/ranking rejection rules | Task 8 |
| 7. reranking under a COMPLETE run | Task 8 (`model_artifact_candidates_requires_finalized_matching_candidate` allows a second artifact write path; add an explicit test in Task 8 if the executor finds this under-covered) |
| 8. post-membership mutation against pinned runs | Task 6, Task 15 |
| 9. fresh/idempotent/004-to-005 upgrade + invalidation | Task 9, Task 13 |
| 10. five contiguous checksums + validator checksum | Task 10, `test_migrations_v17.py` (pre-existing) |
| 11. exact published SHA-256 constants | Not yet covered — add a small pinned-constant test once the constants are frozen after review, per spec section 6 item 11 and section 9 (Acceptance) |
| 12. deterministic invalidation + later supersession | Task 9 |
| 13. active-selection chain zero-result on invalid leaf | Task 9 |
| 14. interrupted-run recovery | Task 14 |
| 15. complex feature dtypes | Task 11 |
| 16. behavioral Git-ignore checks | Task 12 |
| 17. active-selection query plan + full suites | Task 16, Step 3 |
| 18. wheel build/install verification | Task 17 |
| 19. upgrade rejection scenarios | Task 13, Task 15 (network hash mismatch) |
| 20. unfinalized-but-valid promotion excluded + scale gate | Task 9, Task 16 |

Item 11 (pinning exact published SHA-256 constants) is intentionally left open: the spec says those constants are "frozen after 005 review approval" (section 6, item 11) — i.e., only after a human has actually reviewed and approved the finished migration. Add that pinning test as a final step once this plan's Task 1-16 changes have been reviewed and merged, not before.

- [ ] **Step 5: Report status to the user**

No commit in this task — it's verification-only. Summarize: full suite green, packaging verified, 19/20 spec-section-6 items covered by name, item 11 explicitly deferred pending review sign-off.

---

## Notes for the executing engineer

- Every SQL trigger in this plan follows the existing style in `002_model_integrity.sql`/`003_model_integrity_guards.sql`/`004_training_run_identity.sql` — `RAISE(ABORT, '...')` inside `BEGIN...END`, `WHEN NOT EXISTS(...)` for validate-on-insert, `WHEN NEW.col IS NOT OLD.col` for immutability. Stay consistent with that style for anything not explicitly spelled out here.
- Tasks 4-9 all append to the same file (`005_provenance_integrity.sql`) and are ordered so each part's tests only require the SQL written up to that point — but the file is not a valid, self-consistent migration until Task 9 finishes it (e.g. `model_promotion_invalidations` in Part F references `model_promotions`, which only Part F touches; the DROP/CREATE of `active_model_selections` in Part F must run after every trigger in Parts A-E exists). Do not reorder tasks.
- If any trigger's `WHEN` clause needs adjustment during TDD (a test fails for a subtly different reason than expected — SQLite's `json_each` semantics and generated-column behavior are the most likely sources of surprises), that is expected and is exactly what the RED-GREEN-REFACTOR loop in each task is for. Fix the trigger, not the test, unless the test itself is wrong per the spec.
