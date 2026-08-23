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
