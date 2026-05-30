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
