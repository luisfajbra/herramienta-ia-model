# tests/database/test_migration_005_pinning.py
import hashlib
from pathlib import Path
import shutil

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"

_INP_BYTES = b"inp-bytes"
_NETWORK_SHA256 = hashlib.sha256(_INP_BYTES).hexdigest()


def _seed_pinnable_run(conn):
    conn.execute(
        """
        INSERT INTO networks (network_id, network_sha256, name, source_filename,
                               inp_bytes, flow_units, created_at_utc)
        VALUES (1, ?, 'net', 'net.inp', ?, 'LPS', '2026-08-22T00:00:00+00:00')
        """,
        (_NETWORK_SHA256, _INP_BYTES),
    )
    conn.execute(
        "INSERT INTO nodes (node_pk, network_id, node_id, node_type) "
        "VALUES (1, 1, 'N1', 'JUNCTION')"
    )
    conn.execute(
        """
        INSERT INTO scenarios (scenario_id, network_id, scenario_key, scenario_kind,
                                factor_mult, shape_id, duracion_horas,
                                tiempo_al_pico_h, config_json, config_sha256)
        VALUES (1, 1, 'baseline', 'baseline', NULL, NULL, 1.0, 0.5, '{}', ?)
        """,
        ("e" * 64,),
    )
    conn.execute(
        """
        INSERT INTO runs (run_id, network_id, scenario_id, status,
                           config_sha256, node_count)
        VALUES (1, 1, 1, 'COMPLETE', ?, 1)
        """,
        ("d" * 64,),
    )
    conn.execute(
        "INSERT INTO node_results (run_id, network_id, node_pk, inunda, vol_inundacion_m3) "
        "VALUES (1, 1, 1, 0, 0.0)"
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
    conn = connect_managed_database(tmp_path / "db.sqlite3")
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


def test_pinned_scenario_row_cannot_be_updated_or_deleted(tmp_path):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    for name in (
        "001_v17_initial.sql", "002_model_integrity.sql",
        "003_model_integrity_guards.sql", "004_training_run_identity.sql",
        "005_provenance_integrity.sql",
    ):
        shutil.copyfile(SQL_DIR / name, catalog / name)
    conn = connect_managed_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_pinnable_run(conn)
    conn.execute("INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)")
    conn.commit()

    # Reproduces the reported bug scenario exactly: updating config_json and
    # factor_mult together on a scenario referenced by a pinned run.
    with pytest.raises(Exception):
        conn.execute(
            "UPDATE scenarios SET config_json = '{\"changed\": true}', "
            "factor_mult = 2.0 WHERE scenario_id = 1"
        )
    conn.rollback()
    with pytest.raises(Exception):
        conn.execute("DELETE FROM scenarios WHERE scenario_id = 1")
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
    conn = connect_managed_database(tmp_path / "db.sqlite3")
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
