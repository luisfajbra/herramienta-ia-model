import sqlite3
import uuid
from pathlib import Path

import pytest

from swmm_resilience.database.schema import create_schema
from swmm_resilience.database.queries import register_temporal_artifact
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
