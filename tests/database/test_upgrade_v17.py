# tests/database/test_upgrade_v17.py
from pathlib import Path

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.upgrade import upgrade_database_with_backup
from swmm_resilience.database.workflow_lock import WorkflowLock, WorkflowLockError


def test_upgrade_backs_up_before_applying_005(tmp_path):
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


def test_upgrade_applies_all_migrations_from_a_fresh_database(tmp_path):
    db_path = tmp_path / "fresh.sqlite3"
    receipt = upgrade_database_with_backup(db_path, tmp_path / "backups")
    assert receipt.schema_version_before == 0

    verify = connect_database(db_path)
    applied = [row[0] for row in verify.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()]
    assert applied == [1, 2, 3, 4, 5]
    verify.close()


def test_upgrade_fails_when_workflow_lock_already_held(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    conn = connect_database(db_path)
    apply_migrations(conn)
    conn.close()

    with WorkflowLock(db_path):
        with pytest.raises(WorkflowLockError):
            upgrade_database_with_backup(db_path, tmp_path / "backups")
