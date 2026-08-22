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
