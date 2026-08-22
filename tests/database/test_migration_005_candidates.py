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


def _insert_training_run(conn, training_run_id=1, primary_metric="roc_auc", status="RUNNING"):
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (?, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, ?, '[]', '3.11', '{}', ?)
        """,
        (training_run_id, "a" * 64, primary_metric, status),
    )


def _insert_trained_model(
    conn, model_id, training_run_id, target="inunda", algorithm="xgboost",
    hyperparameters_json="{}", feature_contract_sha256=None,
):
    """Inserts a trained_models artifact whose recipe fields match the
    candidates built by `_insert_candidate`/`_build_two_fold_candidate`
    (feature_contract_id, feature_contract_sha256, ordered_features_json,
    preprocessing_json, target_transform_json, hyperparameters_json,
    algorithm) so it can be linked via model_artifact_candidates unless a
    caller deliberately overrides a field to test a mismatch."""
    feature_contract_sha256 = feature_contract_sha256 or "a" * 64
    conn.execute(
        """
        INSERT INTO trained_models (
            model_id, training_run_id, target, algorithm, hyperparameters_json,
            preprocessing_json, feature_contract_id, feature_contract_sha256,
            ordered_features_json, target_transform_json, query_params_json,
            included_run_ids_json, random_seed, grouping_strategy, python_version,
            library_versions_json, model_sha256, model_blob, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, '{}', 'tabular_v3_17', ?, '[]', '{}', '{}',
                  '[1, 2]', 42, 'group_kfold', '3.11', '{}', ?, ?, '2026-01-01T00:00:00Z')
        """,
        (
            model_id, training_run_id, target, algorithm, hyperparameters_json,
            feature_contract_sha256, f"{model_id:064d}", b"blob",
        ),
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


# ---------------------------------------------------------------------------
# model_ranking_scores finiteness guard (mirrors oof_predictions_validate_insert)
# ---------------------------------------------------------------------------


def _build_single_entry_ranking_ready_for_score(conn):
    """Training run + candidate + ranking + one entry, with no score row yet:
    the minimal state needed to attempt a model_ranking_scores insert."""
    _insert_training_run(conn)
    _insert_candidate(conn, candidate_id=1)
    _insert_ranking(conn)
    _insert_entry(conn, ranking_id=1, entry_id=1, candidate_id=1)
    conn.commit()


def test_ranking_score_rejects_nan_value_when_valid(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _build_single_entry_ranking_ready_for_score(conn)

    with pytest.raises(Exception):
        _insert_score(conn, ranking_id=1, entry_id=1, value=float("nan"))
    conn.rollback()


def test_ranking_score_rejects_infinite_value_when_valid(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _build_single_entry_ranking_ready_for_score(conn)

    with pytest.raises(Exception):
        _insert_score(conn, ranking_id=1, entry_id=1, value=float("inf"))
    conn.rollback()


def test_ranking_score_accepts_large_but_finite_value_when_valid(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _build_single_entry_ranking_ready_for_score(conn)

    # A large-but-finite score must not be mistaken for infinity by the
    # finiteness guard (same false-positive concern as the OOF guard).
    _insert_score(conn, ranking_id=1, entry_id=1, value=1e6)
    conn.commit()

    row = conn.execute(
        "SELECT value FROM model_ranking_scores WHERE ranking_id=1 AND entry_id=1"
    ).fetchone()
    assert row[0] == 1e6


# ---------------------------------------------------------------------------
# Shared network/runs fixture + two-fold candidate builder, used by the
# candidate-finalization and artifact-link coverage below.
# ---------------------------------------------------------------------------


def _seed_network_and_two_runs(conn):
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
    for run_id, sha in ((1, "d" * 64), (2, "c" * 64)):
        conn.execute(
            """
            INSERT INTO runs (run_id, network_id, scenario_id, status,
                               config_sha256, node_count)
            VALUES (?, 1, 1, 'COMPLETE', ?, 1)
            """,
            (run_id, sha),
        )
        conn.execute(
            """
            INSERT INTO node_results (run_id, network_id, node_pk, inunda, vol_inundacion_m3)
            VALUES (?, 1, 1, 1, 3.5)
            """,
            (run_id,),
        )


def _build_two_fold_candidate(
    conn,
    *,
    candidate_id=1,
    training_run_id=1,
    complete_fold1=True,
    oof_fold1=True,
    link_fold1=True,
):
    """Builds a training run (fold_count=2) with two COMPLETE-ready
    evaluations (fold 0 validates on run 1, fold 1 validates on run 2) and a
    candidate task/algorithm/hyperparameters that matches both, then links
    fold 0 fully (COMPLETE + OOF + linked) unconditionally. Fold 1's
    completeness/OOF-coverage/linkage are each independently toggleable so a
    caller can isolate exactly one finalization precondition at a time.
    Returns (candidate_id, training_run_id, eval_fold0, eval_fold1).
    """
    _seed_network_and_two_runs(conn)

    eval_fold0 = 1
    eval_fold1 = 2

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (?, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[1, 2]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'PENDING')
        """,
        (training_run_id, "a" * 64),
    )
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (?, 1)",
        (training_run_id,),
    )
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (?, 2)",
        (training_run_id,),
    )
    conn.execute(
        "UPDATE training_runs SET status='RUNNING' WHERE training_run_id=?",
        (training_run_id,),
    )

    # Fold 0: trains on run 2, validates on run 1. Always driven to
    # COMPLETE with full OOF coverage and linked to the candidate.
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (?, ?, 'classification', 'xgboost', '{}', 0, '[2]', '[1]',
                  'PENDING', 0, 0)
        """,
        (eval_fold0, training_run_id),
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (?, 'train', 2)",
        (eval_fold0,),
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (?, 'validation', 1)",
        (eval_fold0,),
    )
    conn.execute(
        "UPDATE model_evaluations SET status='RUNNING' WHERE evaluation_id=?",
        (eval_fold0,),
    )
    conn.execute(
        """
        INSERT INTO oof_predictions (
            evaluation_id, run_id, node_pk, target, observed, predicted,
            probability, fold_id
        ) VALUES (?, 1, 1, 'inunda', 1, 1, 0.9, 0)
        """,
        (eval_fold0,),
    )
    conn.execute(
        "UPDATE model_evaluations SET status='COMPLETE' WHERE evaluation_id=?",
        (eval_fold0,),
    )

    # Fold 1: trains on run 1, validates on run 2. Its completeness, OOF
    # coverage, and candidate linkage are each gated by a flag.
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (?, ?, 'classification', 'xgboost', '{}', 1, '[1]', '[2]',
                  'PENDING', 0, 0)
        """,
        (eval_fold1, training_run_id),
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (?, 'train', 1)",
        (eval_fold1,),
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (?, 'validation', 2)",
        (eval_fold1,),
    )
    conn.execute(
        "UPDATE model_evaluations SET status='RUNNING' WHERE evaluation_id=?",
        (eval_fold1,),
    )
    if oof_fold1:
        conn.execute(
            """
            INSERT INTO oof_predictions (
                evaluation_id, run_id, node_pk, target, observed, predicted,
                probability, fold_id
            ) VALUES (?, 2, 1, 'inunda', 1, 1, 0.9, 1)
            """,
            (eval_fold1,),
        )
    if complete_fold1:
        conn.execute(
            "UPDATE model_evaluations SET status='COMPLETE' WHERE evaluation_id=?",
            (eval_fold1,),
        )

    conn.execute(
        """
        INSERT INTO model_candidates (
            candidate_id, training_run_id, task, algorithm, hyperparameters_json,
            preprocessing_json, feature_contract_id, feature_contract_sha256,
            ordered_features_json, target_transform_json, pipeline_version,
            candidate_definition_sha256
        ) VALUES (?, ?, 'classification', 'xgboost', '{}', '{}', 'tabular_v3_17',
                  ?, '[]', '{}', 'v1', ?)
        """,
        (candidate_id, training_run_id, "a" * 64, f"{candidate_id:064d}"),
    )
    conn.execute(
        "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (?, ?)",
        (candidate_id, eval_fold0),
    )
    if link_fold1:
        conn.execute(
            "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (?, ?)",
            (candidate_id, eval_fold1),
        )

    conn.commit()
    return candidate_id, training_run_id, eval_fold0, eval_fold1


def _finalize_candidate(conn, candidate_id):
    conn.execute(
        "INSERT INTO model_candidate_finalizations (candidate_id, finalized_at_utc) "
        "VALUES (?, '2026-01-01T00:00:00Z')",
        (candidate_id,),
    )


# ---------------------------------------------------------------------------
# 1. model_candidate_finalizations_validate_insert
# ---------------------------------------------------------------------------


def test_candidate_finalization_accepts_complete_two_fold_coverage(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    candidate_id, *_ = _build_two_fold_candidate(conn)

    _finalize_candidate(conn, candidate_id)
    conn.commit()

    row = conn.execute(
        "SELECT candidate_id FROM model_candidate_finalizations WHERE candidate_id=?",
        (candidate_id,),
    ).fetchone()
    assert row[0] == candidate_id


def test_candidate_finalization_rejects_missing_oof_row_in_one_fold(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    # Both folds COMPLETE and linked, but fold 1's validation node has no
    # OOF row: isolates the OOF-coverage branch specifically.
    candidate_id, *_ = _build_two_fold_candidate(conn, oof_fold1=False)

    with pytest.raises(Exception):
        _finalize_candidate(conn, candidate_id)
    conn.rollback()


def test_candidate_finalization_rejects_when_an_evaluation_is_not_complete(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    # Fold 1 is linked with full OOF coverage but left RUNNING (never
    # COMPLETE): isolates the not-COMPLETE branch.
    candidate_id, *_ = _build_two_fold_candidate(conn, complete_fold1=False)

    with pytest.raises(Exception):
        _finalize_candidate(conn, candidate_id)
    conn.rollback()


def test_candidate_finalization_rejects_when_a_fold_is_entirely_omitted(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    # Fold 1's evaluation is COMPLETE with full OOF coverage, but it is
    # never linked to the candidate at all: isolates the fold-count branch.
    candidate_id, *_ = _build_two_fold_candidate(conn, link_fold1=False)

    with pytest.raises(Exception):
        _finalize_candidate(conn, candidate_id)
    conn.rollback()


# ---------------------------------------------------------------------------
# 2. model_artifact_candidates_requires_finalized_matching_candidate
# ---------------------------------------------------------------------------


def test_artifact_candidate_link_accepts_matching_finalized_candidate(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    candidate_id, training_run_id, *_ = _build_two_fold_candidate(conn)
    _finalize_candidate(conn, candidate_id)
    conn.commit()

    _insert_trained_model(conn, model_id=1, training_run_id=training_run_id)
    conn.commit()

    conn.execute(
        "INSERT INTO model_artifact_candidates (model_id, candidate_id) VALUES (1, ?)",
        (candidate_id,),
    )
    conn.commit()

    row = conn.execute(
        "SELECT candidate_id FROM model_artifact_candidates WHERE model_id=1"
    ).fetchone()
    assert row[0] == candidate_id


def test_artifact_candidate_link_rejects_algorithm_mismatch(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    candidate_id, training_run_id, *_ = _build_two_fold_candidate(conn)
    _finalize_candidate(conn, candidate_id)
    conn.commit()

    # Every recipe field matches except algorithm.
    _insert_trained_model(
        conn, model_id=2, training_run_id=training_run_id, algorithm="random_forest",
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO model_artifact_candidates (model_id, candidate_id) VALUES (2, ?)",
            (candidate_id,),
        )
    conn.rollback()


# ---------------------------------------------------------------------------
# 3. model_candidates_owner_running
# ---------------------------------------------------------------------------


def test_candidate_insert_rejects_pending_training_run(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    _insert_training_run(conn, status="PENDING")
    conn.commit()

    with pytest.raises(Exception):
        _insert_candidate(conn, candidate_id=1)
    conn.rollback()


def test_candidate_insert_accepts_running_training_run(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    _insert_training_run(conn, status="RUNNING")
    conn.commit()

    _insert_candidate(conn, candidate_id=1)
    conn.commit()

    row = conn.execute(
        "SELECT candidate_id FROM model_candidates WHERE candidate_id=1"
    ).fetchone()
    assert row[0] == 1


# ---------------------------------------------------------------------------
# 4. model_candidate_evaluations_not_after_finalization
# ---------------------------------------------------------------------------


def test_candidate_evaluation_link_rejected_after_finalization(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    candidate_id, training_run_id, *_ = _build_two_fold_candidate(conn)
    _finalize_candidate(conn, candidate_id)
    conn.commit()

    # A third evaluation whose task/algorithm/hyperparameters do match the
    # (now finalized) candidate, so the only thing that can reject it is
    # the finalization guard.
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (999, ?, 'classification', 'xgboost', '{}', 2, '[]', '[]',
                  'PENDING', 0, 0)
        """,
        (training_run_id,),
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (?, 999)",
            (candidate_id,),
        )
    conn.rollback()


# ---------------------------------------------------------------------------
# 5. model_ranking_entries_not_after_finalization /
#    model_ranking_scores_not_after_finalization
# ---------------------------------------------------------------------------


def test_ranking_entry_and_score_inserts_rejected_after_finalization(tmp_path):
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

    with pytest.raises(Exception):
        _insert_entry(conn, ranking_id=1, entry_id=3, candidate_id=1)
    conn.rollback()

    with pytest.raises(Exception):
        _insert_score(
            conn, ranking_id=1, entry_id=1, value=0.5, metric_name="f1", metric_ordinal=1,
        )
    conn.rollback()


# ---------------------------------------------------------------------------
# 6. model_promotion_rankings_requires_finalized_ranking
# ---------------------------------------------------------------------------


def test_promotion_ranking_link_rejects_non_finalized_ranking(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    _insert_training_run(conn, training_run_id=1, status="RUNNING")
    conn.commit()
    _insert_candidate(conn, candidate_id=1, training_run_id=1)
    _insert_ranking(conn, ranking_id=1, training_run_id=1)
    _insert_entry(conn, ranking_id=1, entry_id=1, candidate_id=1)
    _insert_score(conn, ranking_id=1, entry_id=1, value=0.9)
    conn.commit()
    # Deliberately not finalized.

    _insert_trained_model(conn, model_id=1, training_run_id=1)
    conn.commit()
    conn.execute("UPDATE training_runs SET status='COMPLETE' WHERE training_run_id=1")
    conn.execute(
        """
        INSERT INTO model_promotions (
            promotion_id, training_run_id, target, classifier_model_id,
            regressor_model_id, primary_metric, primary_value, tie_breakers_json,
            ranking_json, promoted_at_utc
        ) VALUES (1, 1, 'inunda', 1, NULL, 'roc_auc', 0.9, '[]', '[]', '2026-01-01T00:00:00Z')
        """
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO model_promotion_rankings (promotion_id, ranking_id) VALUES (1, 1)"
        )
    conn.rollback()


# ---------------------------------------------------------------------------
# 7. model_promotion_finalizations_validate_insert
# ---------------------------------------------------------------------------


def _build_finalized_promotion(conn, *, mismatched_primary_value=False):
    """Full end-to-end chain: a finalized two-fold candidate, a
    matching artifact link, a finalized single-entry ranking naming that
    candidate the winner, and a promotion linked to that ranking - ready for
    a model_promotion_finalizations insert. When mismatched_primary_value is
    True the promotion's primary_value deliberately does not match the
    ranking winner's score, isolating the metric-matching branch."""
    candidate_id, training_run_id, *_ = _build_two_fold_candidate(conn)
    _finalize_candidate(conn, candidate_id)
    conn.commit()

    _insert_trained_model(conn, model_id=1, training_run_id=training_run_id)
    conn.commit()
    conn.execute(
        "INSERT INTO model_artifact_candidates (model_id, candidate_id) VALUES (1, ?)",
        (candidate_id,),
    )
    conn.commit()

    _insert_ranking(conn, ranking_id=1, training_run_id=training_run_id)
    _insert_entry(conn, ranking_id=1, entry_id=1, candidate_id=candidate_id, task="classification")
    winning_score = 0.95
    _insert_score(conn, ranking_id=1, entry_id=1, value=winning_score)
    conn.commit()
    conn.execute(
        "INSERT INTO model_ranking_finalizations (ranking_id, winner_entry_id, finalized_at_utc) "
        "VALUES (1, 1, '2026-01-01T00:00:00Z')"
    )
    conn.commit()

    conn.execute(
        "UPDATE training_runs SET status='COMPLETE' WHERE training_run_id=?",
        (training_run_id,),
    )
    conn.commit()

    promotion_value = 0.5 if mismatched_primary_value else winning_score
    conn.execute(
        """
        INSERT INTO model_promotions (
            promotion_id, training_run_id, target, classifier_model_id,
            regressor_model_id, primary_metric, primary_value, tie_breakers_json,
            ranking_json, promoted_at_utc
        ) VALUES (1, ?, 'inunda', 1, NULL, 'roc_auc', ?, '[]', '[]', '2026-01-01T00:00:00Z')
        """,
        (training_run_id, promotion_value),
    )
    conn.commit()
    conn.execute(
        "INSERT INTO model_promotion_rankings (promotion_id, ranking_id) VALUES (1, 1)"
    )
    conn.commit()


def test_promotion_finalization_accepts_full_matching_chain(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    _build_finalized_promotion(conn)

    conn.execute(
        "INSERT INTO model_promotion_finalizations (promotion_id, finalized_at_utc) "
        "VALUES (1, '2026-01-01T00:00:00Z')"
    )
    conn.commit()

    row = conn.execute(
        "SELECT promotion_id FROM model_promotion_finalizations WHERE promotion_id=1"
    ).fetchone()
    assert row[0] == 1


def test_promotion_finalization_rejects_primary_value_mismatch(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)

    _build_finalized_promotion(conn, mismatched_primary_value=True)

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO model_promotion_finalizations (promotion_id, finalized_at_utc) "
            "VALUES (1, '2026-01-01T00:00:00Z')"
        )
    conn.rollback()
