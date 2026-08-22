# tests/database/test_migration_005_candidates.py
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


def test_candidate_tables_exist_and_are_append_only(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    for table in (
        "model_candidates",
        "model_candidate_evaluations",
        "model_candidate_finalizations",
        "model_artifact_candidates",
        "model_rankings",
        "model_ranking_entries",
        "model_ranking_scores",
        "model_ranking_finalizations",
        "model_promotion_rankings",
        "model_promotion_finalizations",
    ):
        conn.execute(f"SELECT * FROM {table} LIMIT 0")  # raises if table is missing


def test_candidate_evaluation_link_requires_matching_task_and_algorithm(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'RUNNING')
        """,
        ("a" * 64,),
    )
    conn.execute(
        """
        INSERT INTO model_candidates (
            candidate_id, training_run_id, task, algorithm, hyperparameters_json,
            preprocessing_json, feature_contract_id, feature_contract_sha256,
            ordered_features_json, target_transform_json, pipeline_version,
            candidate_definition_sha256
        ) VALUES (1, 1, 'classification', 'xgboost', '{}', '{}', 'tabular_v3_17',
                  ?, '[]', '{}', 'v1', ?)
        """,
        ("a" * 64, "c" * 64),
    )
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (1, 1, 'regression', 'xgboost', '{}', 0, '[]', '[]', 'PENDING', 0, 0)
        """
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (1, 1)"
        )
    conn.rollback()


def test_evaluation_id_is_unique_across_candidate_links(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'RUNNING')
        """,
        ("a" * 64,),
    )
    for candidate_id in (1, 2):
        conn.execute(
            """
            INSERT INTO model_candidates (
                candidate_id, training_run_id, task, algorithm, hyperparameters_json,
                preprocessing_json, feature_contract_id, feature_contract_sha256,
                ordered_features_json, target_transform_json, pipeline_version,
                candidate_definition_sha256
            ) VALUES (?, 1, 'classification', 'xgboost', '{}', '{}', 'tabular_v3_17',
                      ?, '[]', '{}', 'v1', ?)
            """,
            (candidate_id, "a" * 64, f"{candidate_id:064d}"),
        )
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (1, 1, 'classification', 'xgboost', '{}', 0, '[]', '[]', 'PENDING', 0, 0)
        """
    )
    conn.execute(
        "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (1, 1)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (2, 1)"
        )
    conn.rollback()
