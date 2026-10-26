import sqlite3

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations


@pytest.fixture
def migrated_conn(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_training_samples_v4_21_view_exists(migrated_conn):
    row = migrated_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='training_samples_v4_21'"
    ).fetchone()
    assert row is not None


def test_training_samples_v4_21_view_has_expected_columns(migrated_conn):
    cols = [
        row[1] for row in migrated_conn.execute(
            "PRAGMA table_info(training_samples_v4_21)"
        ).fetchall()
    ]
    assert "volumen_local_integrado_m3" in cols
    assert "duracion_horas" not in cols
    assert "inunda" in cols


def test_migration_006_and_005_migrations_still_apply_cleanly(migrated_conn):
    # Si esta migracion rompiera algo de las anteriores, apply_migrations
    # (ya ejecutado en el fixture) habria lanzado antes de llegar aqui.
    count = migrated_conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='view'"
    ).fetchone()[0]
    assert count >= 2  # al menos training_samples_v17 y training_samples_v4_21
