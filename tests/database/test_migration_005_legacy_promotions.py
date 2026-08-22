# tests/database/test_migration_005_legacy_promotions.py
from pathlib import Path
import shutil

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations

SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def test_every_pre005_promotion_is_invalidated_and_selections_are_empty(tmp_path):
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

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status, completed_at_utc
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}',
                  'COMPLETE', '2026-08-21T00:00:00+00:00')
        """,
        ("a" * 64,),
    )
    conn.execute(
        """
        INSERT INTO trained_models (
            model_id, training_run_id, target, algorithm, hyperparameters_json,
            preprocessing_json, feature_contract_id, feature_contract_sha256,
            ordered_features_json, target_transform_json, query_params_json,
            included_run_ids_json, random_seed, grouping_strategy, python_version,
            library_versions_json, model_sha256, model_blob, created_at_utc
        ) VALUES (1, 1, 'inunda', 'xgboost', '{}', '{}', 'tabular_v3_17', ?, '[]',
                  '{}', '{}', '[]', 42, 'group_kfold', '3.11', '{}', ?, ?,
                  '2026-08-21T00:00:00+00:00')
        """,
        ("b" * 64, "c" * 64, b"blob"),
    )
    conn.execute(
        """
        INSERT INTO model_promotions (
            promotion_id, training_run_id, target, classifier_model_id,
            primary_metric, primary_value, tie_breakers_json, ranking_json,
            promoted_at_utc
        ) VALUES (1, 1, 'inunda', 1, 'roc_auc', 0.9, '[]', '{}',
                  '2026-08-21T00:00:00+00:00')
        """
    )
    conn.execute(
        """
        INSERT INTO model_selections (
            selection_id, target, promotion_id, supersedes_selection_id, selected_at_utc
        ) VALUES (1, 'inunda', 1, NULL, '2026-08-21T00:00:00+00:00')
        """
    )
    conn.commit()

    shutil.copyfile(
        SQL_DIR / "005_provenance_integrity.sql",
        catalog / "005_provenance_integrity.sql",
    )
    apply_migrations(conn, migration_dir=catalog)

    invalidated = conn.execute(
        "SELECT reason FROM model_promotion_invalidations WHERE promotion_id = 1"
    ).fetchone()
    assert invalidated is not None

    active = conn.execute("SELECT * FROM active_model_selections").fetchall()
    assert active == []  # upgraded DB may have no active selection for any target

    valid = conn.execute(
        "SELECT 1 FROM valid_model_promotions WHERE promotion_id = 1"
    ).fetchall()
    assert valid == []
