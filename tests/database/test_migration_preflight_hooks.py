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
    applied = conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    assert [row[0] for row in applied] == [1]
    conn.close()


def test_preflight_hook_passing_allows_migration_to_apply(tmp_path):
    catalog = _catalog(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog, preflight_hooks={2: lambda c: None})
    applied = conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    assert [row[0] for row in applied] == [1, 2]
    conn.close()
