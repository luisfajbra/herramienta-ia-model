# tests/database/test_migration_005_oof.py
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


def _seed(conn):
    conn.execute(
        """
        INSERT INTO networks (network_id, network_sha256, name, source_filename,
                               inp_bytes, flow_units, created_at_utc)
        VALUES (1, ?, 'net', 'net.inp', ?, 'LPS', '2026-08-22T00:00:00+00:00')
        """,
        ("f" * 64, b"inp-bytes"),
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
        "INSERT INTO nodes (node_pk, network_id, node_id, node_type) "
        "VALUES (1, 1, 'N1', 'JUNCTION')"
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
        """
        INSERT INTO runs (run_id, network_id, scenario_id, status,
                           config_sha256, node_count)
        VALUES (2, 1, 1, 'COMPLETE', ?, 1)
        """,
        ("c" * 64,),
    )
    conn.execute(
        """
        INSERT INTO node_results (run_id, network_id, node_pk, inunda, vol_inundacion_m3)
        VALUES (1, 1, 1, 1, 3.5)
        """
    )
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[1, 2]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'PENDING')
        """,
        ("a" * 64,),
    )
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
    )
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 2)"
    )
    conn.execute("UPDATE training_runs SET status='RUNNING' WHERE training_run_id=1")
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (1, 1, 'classification', 'xgboost', '{}', 0, '[2]', '[1]',
                  'PENDING', 0, 0)
        """
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'train', 2)"
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'validation', 1)"
    )
    conn.execute("UPDATE model_evaluations SET status='RUNNING' WHERE evaluation_id=1")
    conn.commit()


def test_oof_insert_requires_running_owner_and_matches_persisted_target(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed(conn)

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO oof_predictions (
                evaluation_id, run_id, node_pk, target, observed, predicted,
                probability, fold_id
            ) VALUES (1, 1, 1, 'inunda', 0, 1, 0.9, 0)
            """
        )
    conn.rollback()

    conn.execute(
        """
        INSERT INTO oof_predictions (
            evaluation_id, run_id, node_pk, target, observed, predicted,
            probability, fold_id
        ) VALUES (1, 1, 1, 'inunda', 1, 1, 0.9, 0)
        """
    )
    conn.commit()


def test_oof_rows_are_append_only(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed(conn)
    conn.execute(
        """
        INSERT INTO oof_predictions (
            evaluation_id, run_id, node_pk, target, observed, predicted,
            probability, fold_id
        ) VALUES (1, 1, 1, 'inunda', 1, 1, 0.9, 0)
        """
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "UPDATE oof_predictions SET predicted = 0 WHERE evaluation_id = 1"
        )
    conn.rollback()

    with pytest.raises(Exception):
        conn.execute("DELETE FROM oof_predictions WHERE evaluation_id = 1")
    conn.rollback()

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT OR REPLACE INTO oof_predictions (
                evaluation_id, run_id, node_pk, target, observed, predicted,
                probability, fold_id
            ) VALUES (1, 1, 1, 'inunda', 1, 0, 0.1, 0)
            """
        )
    conn.rollback()


def test_oof_rejects_out_of_domain_classification_values(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed(conn)

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO oof_predictions (
                evaluation_id, run_id, node_pk, target, observed, predicted,
                probability, fold_id
            ) VALUES (1, 1, 1, 'inunda', 1, 1, 1.5, 0)
            """
        )
    conn.rollback()
