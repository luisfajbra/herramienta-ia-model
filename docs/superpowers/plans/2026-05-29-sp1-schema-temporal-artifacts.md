# SP1 — Schema v2 + Temporal Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `temporal_artifacts` SQLite table and the `input_source` column to `runs`, then wire `main.py` to register each Parquet file in the database after saving it.

**Architecture:** Schema changes are handled by extending `SCHEMA_SQL` and `REQUIRED_COLUMNS` in `schema.py` (using the existing migration mechanism). A new `queries.py` module holds the `register_temporal_artifact()` function so that SQL logic stays out of `main.py`. `reset.py` is updated so that `--db` clears `temporal_artifacts` before `runs`.

**Tech Stack:** Python 3.10+, SQLite (stdlib `sqlite3`), pytest 7+

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Create | `pytest.ini` | Tell pytest where tests live and add project root to path |
| Create | `tests/__init__.py` | Make `tests/` a package |
| Create | `tests/database/__init__.py` | Make `tests/database/` a package |
| Create | `tests/database/test_temporal_artifacts.py` | All tests for this SP |
| Modify | `swmm_resilience/database/schema.py` | Add `temporal_artifacts` table + `input_source` column |
| Create | `swmm_resilience/database/queries.py` | `register_temporal_artifact()` |
| Modify | `swmm_resilience/reset.py` | Add `temporal_artifacts` to reset order |
| Modify | `swmm_resilience/main.py` | Call `register_temporal_artifact()` + set `input_source` on INSERT |

---

## Task 1: Test infrastructure

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/database/__init__.py`

- [ ] **Step 1.1: Create `pytest.ini` at the project root**

```ini
[pytest]
pythonpath = .
testpaths = tests
addopts = -v
```

- [ ] **Step 1.2: Create the test package directories**

```bash
mkdir -p tests/database
touch tests/__init__.py tests/database/__init__.py
```

- [ ] **Step 1.3: Verify pytest can be imported**

Run: `python -m pytest --collect-only 2>&1 | head -5`
Expected: `no tests ran` (or similar — no ImportError)

---

## Task 2: Schema — `temporal_artifacts` table + `input_source` column (TDD)

**Files:**
- Create: `tests/database/test_temporal_artifacts.py`
- Modify: `swmm_resilience/database/schema.py`

### Step 2.1: Write failing tests for schema

- [ ] **Create `tests/database/test_temporal_artifacts.py` with the schema tests:**

```python
import sqlite3
import uuid
from pathlib import Path

import pytest

from swmm_resilience.database.schema import create_schema
from swmm_resilience.reset import reset_db


# ── helpers ───────────────────────────────────────────────────────────────────

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in cols


def _insert_run(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        """INSERT INTO runs
               (run_id, network_file, network_hash, scenario_type,
                spatial_pattern, delta_inflow_lps, inflow_multiplier,
                executed_at, status)
           VALUES (?, 'net.inp', 'abc123', 'steady', 'uniform',
                   1.0, 1.0, '2026-01-01T00:00:00', 'completed')""",
        (run_id,),
    )
    conn.commit()


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    yield conn
    conn.close()


# ── schema tests ──────────────────────────────────────────────────────────────

class TestTemporalArtifactsTable:
    def test_create_schema_creates_temporal_artifacts_table(self, db):
        assert _table_exists(db, "temporal_artifacts")

    def test_migrate_adds_table_to_existing_db(self):
        """create_schema on a DB that already exists but lacks the table."""
        conn = sqlite3.connect(":memory:")
        create_schema(conn)
        # Simulate an "old" DB by removing the table
        conn.execute("DROP TABLE IF EXISTS temporal_artifacts")
        conn.commit()
        _insert_run(conn, "run-preserve-me")
        # Re-run schema — should recreate table without losing other data
        create_schema(conn)
        assert _table_exists(conn, "temporal_artifacts")
        row = conn.execute(
            "SELECT run_id FROM runs WHERE run_id='run-preserve-me'"
        ).fetchone()
        assert row is not None, "Existing run was lost during migration"
        conn.close()


class TestInputSourceColumn:
    def test_runs_has_input_source_column(self, db):
        assert _column_exists(db, "runs", "input_source")

    def test_input_source_defaults_to_steady(self, db):
        _insert_run(db, "run-default-check")
        row = db.execute(
            "SELECT input_source FROM runs WHERE run_id='run-default-check'"
        ).fetchone()
        assert row[0] == "steady"
```

- [ ] **Step 2.2: Run the tests — verify they FAIL**

```bash
python -m pytest tests/database/test_temporal_artifacts.py -v
```

Expected: 4 failures — `temporal_artifacts` table doesn't exist yet, `input_source` column missing.

### Step 2.3: Implement schema changes

- [ ] **Open `swmm_resilience/database/schema.py` and add `temporal_artifacts` to the end of `SCHEMA_SQL` (after the `run_summary` block, before the closing `"""`)**

Find the `SCHEMA_SQL` string (currently ends after `run_summary`). Add this block inside the triple-quoted string, right before the closing `"""`:

```python
CREATE TABLE IF NOT EXISTS temporal_artifacts (
    artifact_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    network_hash    TEXT NOT NULL,
    parquet_path    TEXT NOT NULL,
    node_count      INTEGER NOT NULL,
    step_count      INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
```

- [ ] **In the same file, add `input_source` to `REQUIRED_COLUMNS["runs"]`**

Find the `REQUIRED_COLUMNS` dict. The `"runs"` entry currently is:

```python
"runs": {
    "inflow_multiplier": "REAL NOT NULL DEFAULT 1"
},
```

Change it to:

```python
"runs": {
    "inflow_multiplier": "REAL NOT NULL DEFAULT 1",
    "input_source": "TEXT NOT NULL DEFAULT 'steady'",
},
```

- [ ] **Step 2.4: Run the tests — verify they PASS**

```bash
python -m pytest tests/database/test_temporal_artifacts.py::TestTemporalArtifactsTable tests/database/test_temporal_artifacts.py::TestInputSourceColumn -v
```

Expected: 4 tests PASS

- [ ] **Step 2.5: Commit**

```bash
git add swmm_resilience/database/schema.py pytest.ini tests/__init__.py tests/database/__init__.py tests/database/test_temporal_artifacts.py
git commit -m "feat(db): add temporal_artifacts table and input_source column to runs"
```

---

## Task 3: `queries.py` — `register_temporal_artifact` (TDD)

**Files:**
- Modify: `tests/database/test_temporal_artifacts.py`
- Create: `swmm_resilience/database/queries.py`

### Step 3.1: Add failing tests for `register_temporal_artifact`

- [ ] **Append these two test classes to `tests/database/test_temporal_artifacts.py`**

Add these classes at the bottom of the existing file (after `TestInputSourceColumn`):

```python
from swmm_resilience.database.queries import register_temporal_artifact


class TestRegisterTemporalArtifact:
    def test_inserts_row(self, db):
        _insert_run(db, "run-001")
        register_temporal_artifact(
            db,
            run_id="run-001",
            network_hash="abc123",
            parquet_path=Path("/data/run_001.parquet"),
            node_count=10,
            step_count=20,
        )
        row = db.execute(
            "SELECT run_id, node_count, step_count FROM temporal_artifacts WHERE run_id='run-001'"
        ).fetchone()
        assert row is not None
        assert row[0] == "run-001"
        assert row[1] == 10
        assert row[2] == 20

    def test_returns_uuid(self, db):
        _insert_run(db, "run-002")
        artifact_id = register_temporal_artifact(
            db,
            run_id="run-002",
            network_hash="abc123",
            parquet_path=Path("/data/run_002.parquet"),
            node_count=5,
            step_count=10,
        )
        uuid.UUID(artifact_id)  # raises ValueError if not a valid UUID4

    def test_parquet_path_stored_as_string(self, db):
        _insert_run(db, "run-003")
        register_temporal_artifact(
            db,
            run_id="run-003",
            network_hash="abc123",
            parquet_path=Path("/data/run_003.parquet"),
            node_count=3,
            step_count=6,
        )
        row = db.execute(
            "SELECT parquet_path FROM temporal_artifacts WHERE run_id='run-003'"
        ).fetchone()
        assert row[0] == "/data/run_003.parquet"
```

Note: also add `from swmm_resilience.database.queries import register_temporal_artifact` to the imports at the top of the test file (after the existing imports).

- [ ] **Step 3.2: Run to verify FAIL**

```bash
python -m pytest tests/database/test_temporal_artifacts.py::TestRegisterTemporalArtifact -v
```

Expected: 3 failures — `queries` module doesn't exist yet.

### Step 3.3: Implement `queries.py`

- [ ] **Create `swmm_resilience/database/queries.py`**

```python
"""
High-level database query helpers — separated from repository.py so
SQL logic is testable in isolation.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..utils import new_id


def register_temporal_artifact(
    conn: sqlite3.Connection,
    run_id: str,
    network_hash: str,
    parquet_path: Path,
    node_count: int,
    step_count: int,
) -> str:
    """Insert a row into temporal_artifacts. Returns the artifact_id."""
    artifact_id = new_id()
    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO temporal_artifacts
            (artifact_id, run_id, network_hash, parquet_path,
             node_count, step_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact_id,
            run_id,
            network_hash,
            str(parquet_path),
            node_count,
            step_count,
            created_at,
        ),
    )
    conn.commit()
    return artifact_id
```

- [ ] **Step 3.4: Run all tests — verify they PASS**

```bash
python -m pytest tests/database/test_temporal_artifacts.py -v
```

Expected: 7 tests PASS (4 schema + 3 queries)

- [ ] **Step 3.5: Commit**

```bash
git add swmm_resilience/database/queries.py tests/database/test_temporal_artifacts.py
git commit -m "feat(db): add register_temporal_artifact to queries.py"
```

---

## Task 4: Update `reset.py` (TDD)

**Files:**
- Modify: `tests/database/test_temporal_artifacts.py`
- Modify: `swmm_resilience/reset.py`

### Step 4.1: Add failing test for reset

- [ ] **Append this test class to `tests/database/test_temporal_artifacts.py`**

```python
class TestResetClearsTemporalArtifacts:
    def test_reset_db_clears_temporal_artifacts(self, tmp_path):
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))
        create_schema(conn)
        _insert_run(conn, "run-reset-me")
        conn.execute(
            """INSERT INTO temporal_artifacts
                   (artifact_id, run_id, network_hash, parquet_path,
                    node_count, step_count, created_at)
               VALUES ('art-001', 'run-reset-me', 'abc123',
                       '/data/run_reset_me.parquet', 5, 10, '2026-01-01T00:00:00')"""
        )
        conn.commit()
        conn.close()

        reset_db(db_file)

        conn2 = sqlite3.connect(str(db_file))
        count = conn2.execute(
            "SELECT COUNT(*) FROM temporal_artifacts"
        ).fetchone()[0]
        conn2.close()
        assert count == 0
```

- [ ] **Step 4.2: Run to verify FAIL**

```bash
python -m pytest tests/database/test_temporal_artifacts.py::TestResetClearsTemporalArtifacts -v
```

Expected: FAIL — `temporal_artifacts` is not in `_DB_TABLES_IN_ORDER` so it won't be deleted, but the table exists and the row will remain.

### Step 4.3: Update `reset.py`

- [ ] **Open `swmm_resilience/reset.py`. Find `_DB_TABLES_IN_ORDER` and add `"temporal_artifacts"` as the first entry:**

Current:
```python
_DB_TABLES_IN_ORDER = [
    "node_results",
    "link_results",
    "run_inputs",
    "run_summary",
    "network_nodes",
    "network_links",
    "runs",
]
```

Change to:
```python
_DB_TABLES_IN_ORDER = [
    "temporal_artifacts",
    "node_results",
    "link_results",
    "run_inputs",
    "run_summary",
    "network_nodes",
    "network_links",
    "runs",
]
```

- [ ] **Step 4.4: Run all tests — verify they PASS**

```bash
python -m pytest tests/database/test_temporal_artifacts.py -v
```

Expected: 8 tests PASS

- [ ] **Step 4.5: Commit**

```bash
git add swmm_resilience/reset.py tests/database/test_temporal_artifacts.py
git commit -m "fix(reset): clear temporal_artifacts when --db flag is used"
```

---

## Task 5: Update `main.py` — set `input_source` and call `register_temporal_artifact`

**Files:**
- Modify: `swmm_resilience/main.py`

This task modifies the simulation loop. There are no unit tests here because testing it requires a full SWMM run. Correctness is verified by running the experiment manually and inspecting the DB.

### Step 5.1: Add import for `register_temporal_artifact`

- [ ] **In `swmm_resilience/main.py`, find the existing imports block (around line 27) and add:**

```python
from .database.queries import register_temporal_artifact
```

Place it right after the existing `from .database.repository import (...)` block.

### Step 5.2: Update the `INSERT INTO runs` statement to include `input_source`

- [ ] **Find the `conn.execute(...)` call that inserts into `runs` (around line 221). The current statement is:**

```python
conn.execute(
    """
    INSERT INTO runs
      (run_id, network_file, network_hash, scenario_type,
       spatial_pattern, delta_inflow_lps, inflow_multiplier, executed_at, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), 'running')
    """,
    (
        run_id,
        network_display_name(inp_path),
        network_hash,
        scenario_type,
        spatial_pattern,
        inflow_multiplier,
        inflow_multiplier,
    ),
)
```

**Replace it with:**

```python
conn.execute(
    """
    INSERT INTO runs
      (run_id, network_file, network_hash, scenario_type,
       spatial_pattern, delta_inflow_lps, inflow_multiplier,
       input_source, executed_at, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 'running')
    """,
    (
        run_id,
        network_display_name(inp_path),
        network_hash,
        scenario_type,
        spatial_pattern,
        inflow_multiplier,
        inflow_multiplier,
        "hydrograph" if scenario_mode == SCENARIO_MODE_TIMESERIES else "steady",
    ),
)
```

### Step 5.3: Add `register_temporal_artifact` call after Parquet save

- [ ] **Find the block that saves the Parquet (around line 255). The current code is:**

```python
node_timeseries_path = (
    network_node_timeseries_dir(inp_path) / f"run_{run_id}.parquet"
)
save_node_timeseries_parquet(
    results["node_timeseries_records"],
    node_timeseries_path,
)
update_run_status(conn, run_id, "completed")
```

**Replace it with:**

```python
node_timeseries_path = (
    network_node_timeseries_dir(inp_path) / f"run_{run_id}.parquet"
)
ts_records = results["node_timeseries_records"]
save_node_timeseries_parquet(ts_records, node_timeseries_path)

if ts_records:
    _node_ids = {r["node_id"] for r in ts_records}
    _node_count = len(_node_ids)
    _step_count = len(ts_records) // _node_count if _node_count else 0
else:
    _node_count = 0
    _step_count = 0

register_temporal_artifact(
    conn,
    run_id=run_id,
    network_hash=network_hash,
    parquet_path=node_timeseries_path,
    node_count=_node_count,
    step_count=_step_count,
)
update_run_status(conn, run_id, "completed")
```

### Step 5.4: Smoke-test with the DB viewer

- [ ] **Run a minimal simulation (1 multiplier) and confirm the DB has the new row:**

```bash
python -m swmm_resilience.main --multipliers 1.0 --scenario-mode steady
```

If the full experiment is slow, open a Python REPL instead:

```python
import sqlite3
from swmm_resilience.config import DEFAULT_DB_FILE
conn = sqlite3.connect(DEFAULT_DB_FILE)
print(conn.execute("SELECT * FROM temporal_artifacts LIMIT 3").fetchall())
print(conn.execute("SELECT run_id, input_source FROM runs LIMIT 3").fetchall())
conn.close()
```

Expected: at least one row in `temporal_artifacts` and `input_source` populated in `runs`.

- [ ] **Step 5.5: Run the full test suite to confirm nothing regressed**

```bash
python -m pytest tests/ -v
```

Expected: all 8 tests PASS, 0 failures.

- [ ] **Step 5.6: Commit**

```bash
git add swmm_resilience/main.py
git commit -m "feat(main): register temporal artifacts in DB after Parquet save"
```

---

## Self-review checklist

- [x] Spec requirement: `temporal_artifacts` table → Task 2
- [x] Spec requirement: `input_source` column on `runs` → Task 2
- [x] Spec requirement: `queries.py` with `register_temporal_artifact` → Task 3
- [x] Spec requirement: `reset.py` clears `temporal_artifacts` → Task 4
- [x] Spec requirement: `main.py` calls `register_temporal_artifact` after Parquet save → Task 5
- [x] Spec requirement: `main.py` sets `input_source` to `'hydrograph'` for timeseries runs → Task 5
- [x] Spec interface toward SP2: `input_source='hydrograph'` filter will work after Task 5
- [x] No placeholders or TBDs in any step
- [x] All test code is complete and runnable
- [x] `artifact_id` is a UUID4 (via `new_id()` from utils.py)
- [x] `node_count`/`step_count` computed from `results["node_timeseries_records"]`
