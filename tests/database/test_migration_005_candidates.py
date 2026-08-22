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


def _insert_training_run(conn, training_run_id=1, primary_metric="roc_auc"):
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (?, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, ?, '[]', '3.11', '{}', 'RUNNING')
        """,
        (training_run_id, "a" * 64, primary_metric),
    )


def _insert_candidate(conn, candidate_id, training_run_id=1, task="classification"):
    conn.execute(
        """
        INSERT INTO model_candidates (
            candidate_id, training_run_id, task, algorithm, hyperparameters_json,
            preprocessing_json, feature_contract_id, feature_contract_sha256,
            ordered_features_json, target_transform_json, pipeline_version,
            candidate_definition_sha256
        ) VALUES (?, ?, ?, 'xgboost', '{}', '{}', 'tabular_v3_17',
                  ?, '[]', '{}', 'v1', ?)
        """,
        (candidate_id, training_run_id, task, "a" * 64, f"{candidate_id:064d}"),
    )


def _insert_ranking(
    conn, ranking_id=1, training_run_id=1, primary_direction="maximize",
    primary_metric="roc_auc",
):
    conn.execute(
        """
        INSERT INTO model_rankings (
            ranking_id, training_run_id, target, primary_metric, primary_direction,
            metric_registry_id, metric_registry_sha256, tie_breakers_json,
            invalid_score_policy, created_at_utc
        ) VALUES (?, ?, 'inunda', ?, ?, 'registry_v1', ?, '[]', 'exclude', '2026-01-01T00:00:00Z')
        """,
        (ranking_id, training_run_id, primary_metric, primary_direction, "b" * 64),
    )


def _insert_entry(conn, ranking_id, entry_id, candidate_id, task="classification"):
    if task == "classification":
        conn.execute(
            """
            INSERT INTO model_ranking_entries (
                ranking_id, entry_id, classifier_candidate_id, regressor_candidate_id
            ) VALUES (?, ?, ?, NULL)
            """,
            (ranking_id, entry_id, candidate_id),
        )
    else:
        conn.execute(
            """
            INSERT INTO model_ranking_entries (
                ranking_id, entry_id, classifier_candidate_id, regressor_candidate_id
            ) VALUES (?, ?, NULL, ?)
            """,
            (ranking_id, entry_id, candidate_id),
        )


def _insert_score(
    conn, ranking_id, entry_id, value, valid=1, metric_name="roc_auc",
    metric_ordinal=0, invalid_reason=None,
):
    conn.execute(
        """
        INSERT INTO model_ranking_scores (
            ranking_id, entry_id, metric_ordinal, metric_name, value, valid, invalid_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ranking_id, entry_id, metric_ordinal, metric_name, value, valid, invalid_reason),
    )


def _build_two_entry_ranking(conn, primary_direction, better_value, worse_value):
    """Builds ranking_id=1 with entry 1 (better score) and entry 2 (worse score)."""
    _insert_training_run(conn)
    _insert_candidate(conn, candidate_id=1)
    _insert_candidate(conn, candidate_id=2)
    _insert_ranking(conn, primary_direction=primary_direction)
    _insert_entry(conn, ranking_id=1, entry_id=1, candidate_id=1)
    _insert_entry(conn, ranking_id=1, entry_id=2, candidate_id=2)
    _insert_score(conn, ranking_id=1, entry_id=1, value=better_value)
    _insert_score(conn, ranking_id=1, entry_id=2, value=worse_value)
    conn.commit()


def test_ranking_finalization_rejects_a_non_winning_entry_maximize(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    # entry 1 = roc_auc 0.95 (best), entry 2 = roc_auc 0.80 (worse), maximize
    _build_two_entry_ranking(conn, "maximize", better_value=0.95, worse_value=0.80)

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO model_ranking_finalizations (
                ranking_id, winner_entry_id, finalized_at_utc
            ) VALUES (1, 2, '2026-01-01T00:00:00Z')
            """
        )
    conn.rollback()


def test_ranking_finalization_accepts_the_true_winning_entry(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    _build_two_entry_ranking(conn, "maximize", better_value=0.95, worse_value=0.80)

    conn.execute(
        """
        INSERT INTO model_ranking_finalizations (
            ranking_id, winner_entry_id, finalized_at_utc
        ) VALUES (1, 1, '2026-01-01T00:00:00Z')
        """
    )
    conn.commit()

    row = conn.execute(
        "SELECT winner_entry_id FROM model_ranking_finalizations WHERE ranking_id=1"
    ).fetchone()
    assert row[0] == 1


def test_ranking_finalization_rejects_a_non_winning_entry_minimize(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    # under minimize, entry 1 (0.10) is best, entry 2 (0.50) is worse
    _build_two_entry_ranking(conn, "minimize", better_value=0.10, worse_value=0.50)

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO model_ranking_finalizations (
                ranking_id, winner_entry_id, finalized_at_utc
            ) VALUES (1, 2, '2026-01-01T00:00:00Z')
            """
        )
    conn.rollback()


def test_ranking_finalization_rejects_winner_missing_primary_score(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    _insert_training_run(conn)
    _insert_candidate(conn, candidate_id=1)
    _insert_candidate(conn, candidate_id=2)
    _insert_ranking(conn, primary_direction="maximize")
    _insert_entry(conn, ranking_id=1, entry_id=1, candidate_id=1)
    _insert_entry(conn, ranking_id=1, entry_id=2, candidate_id=2)
    # entry 2 (the intended winner) has no metric_ordinal=0 score row at all.
    _insert_score(conn, ranking_id=1, entry_id=1, value=0.95)
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO model_ranking_finalizations (
                ranking_id, winner_entry_id, finalized_at_utc
            ) VALUES (1, 2, '2026-01-01T00:00:00Z')
            """
        )
    conn.rollback()


def test_ranking_finalization_rejects_when_any_entry_lacks_a_primary_score(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    _insert_training_run(conn)
    _insert_candidate(conn, candidate_id=1)
    _insert_candidate(conn, candidate_id=2)
    _insert_ranking(conn, primary_direction="maximize")
    _insert_entry(conn, ranking_id=1, entry_id=1, candidate_id=1)
    _insert_entry(conn, ranking_id=1, entry_id=2, candidate_id=2)
    # entry 1 (the intended winner) has a valid, best score of its own...
    _insert_score(conn, ranking_id=1, entry_id=1, value=0.95)
    # ...but entry 2 has no primary score row at all: the ranking is incomplete.
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            """
            INSERT INTO model_ranking_finalizations (
                ranking_id, winner_entry_id, finalized_at_utc
            ) VALUES (1, 1, '2026-01-01T00:00:00Z')
            """
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
