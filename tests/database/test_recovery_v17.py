# tests/database/test_recovery_v17.py
import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.recovery import recover_abandoned_training_run


def _running_training_run_with_evaluations(conn):
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
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (1, 1, 'classification', 'xgboost', '{}', 0, '[]', '[]', 'PENDING', 0, 0)
        """
    )
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (2, 1, 'classification', 'xgboost', '{}', 1, '[]', '[]', 'COMPLETE', 1, 1)
        """
    )
    conn.commit()


def test_recovery_marks_running_run_and_pending_evaluations_failed(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    _running_training_run_with_evaluations(conn)

    recover_abandoned_training_run(conn, training_run_id=1)

    run_status = conn.execute(
        "SELECT status, failure_type FROM training_runs WHERE training_run_id = 1"
    ).fetchone()
    assert run_status[0] == "FAILED"
    assert run_status[1] == "interrupted_run_recovery"

    eval_statuses = dict(conn.execute(
        "SELECT evaluation_id, status FROM model_evaluations WHERE training_run_id = 1"
    ).fetchall())
    assert eval_statuses[1] == "FAILED"   # was PENDING -> now FAILED
    assert eval_statuses[2] == "COMPLETE"  # already-complete evidence is untouched


def test_recovery_sets_completed_at_utc_timestamp(tmp_path):
    # model_evaluations has no completed_at_utc column in the v17 schema
    # (only training_runs does), so only the training run's timestamp is
    # asserted here.
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    _running_training_run_with_evaluations(conn)

    recover_abandoned_training_run(conn, training_run_id=1)

    run_completed_at = conn.execute(
        "SELECT completed_at_utc FROM training_runs WHERE training_run_id = 1"
    ).fetchone()[0]

    assert run_completed_at is not None


def test_recovery_leaves_no_dangling_transaction_on_invalid_run(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)

    with pytest.raises(ValueError):
        recover_abandoned_training_run(conn, training_run_id=999)

    # A subsequent statement on the same connection must succeed immediately
    # (no stuck open transaction / lock from the failed call).
    result = conn.execute("SELECT 1").fetchone()
    assert tuple(result) == (1,)


def test_recovery_rejects_non_running_training_run(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'PENDING')
        """,
        ("a" * 64,),
    )
    conn.commit()

    with pytest.raises(ValueError):
        recover_abandoned_training_run(conn, training_run_id=1)


def test_recovery_rejects_promoted_training_run(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    # model_evaluations_running_requires_complete_membership blocks a
    # PENDING->RUNNING transition when an evaluation has empty train/validation
    # membership, and model_evaluations_immutable_delete forbids deleting rows
    # to work around it. So evaluation 1 is seeded directly as COMPLETE here
    # (INSERT bypasses the status-transition trigger, same as evaluation 2 in
    # the shared fixture) rather than transitioned via UPDATE.
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
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (1, 1, 'classification', 'xgboost', '{}', 0, '[]', '[]', 'COMPLETE', 1, 1)
        """
    )
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (2, 1, 'classification', 'xgboost', '{}', 1, '[]', '[]', 'COMPLETE', 1, 1)
        """
    )
    conn.execute(
        "UPDATE training_runs SET status = 'COMPLETE' WHERE training_run_id = 1"
    )
    conn.commit()

    with pytest.raises(ValueError):
        recover_abandoned_training_run(conn, training_run_id=1)
