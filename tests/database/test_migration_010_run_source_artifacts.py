import sqlite3

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations


def test_migration_011_extends_artifact_registry_with_identity_snapshots(tmp_path):
    conn = connect_managed_database(tmp_path / "migration.sqlite3")
    apply_migrations(conn)

    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(run_source_artifacts)")
    }
    assert columns == {
        "run_id",
        "artifact_role",
        "registered_network_id",
        "registered_scenario_id",
        "resolved_path",
        "sha256",
        "byte_size",
        "source_manifest_sha256",
        "registered_at_utc",
    }
    latest = conn.execute(
        "SELECT version, name FROM schema_migrations ORDER BY version DESC LIMIT 1"
    ).fetchone()
    assert tuple(latest) == (11, "run_source_artifact_identity")
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='run_source_artifacts_legacy_010'"
    ).fetchone() is not None
    conn.close()


def test_migration_010_checks_artifact_fields(tmp_path):
    conn = connect_managed_database(tmp_path / "constraints.sqlite3")
    apply_migrations(conn)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO run_source_artifacts VALUES
               (999, 'unknown', 1, 1, '', 'A', 0, 'bad', '')"""
        )
    conn.rollback()
    conn.close()
