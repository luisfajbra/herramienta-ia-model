from pathlib import Path
import shutil
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations


SQL_DIR = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"


def _migration_catalog(tmp_path, *, include_v2, include_v3=False):
    catalog = tmp_path / "migrations"
    catalog.mkdir()
    shutil.copyfile(SQL_DIR / "001_v17_initial.sql", catalog / "001_v17_initial.sql")
    if include_v2:
        shutil.copyfile(
            SQL_DIR / "002_model_integrity.sql",
            catalog / "002_model_integrity.sql",
        )
    if include_v3:
        shutil.copyfile(
            SQL_DIR / "003_model_integrity_guards.sql",
            catalog / "003_model_integrity_guards.sql",
        )
    return catalog


def _insert_training_run(
    conn,
    training_run_id=1,
    target="system",
    status="COMPLETE",
):
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status, completed_at_utc
        ) VALUES (?, ?, 'tabular_v3_17', ?, 'SELECT 1', '{}', '[]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.13', '{}',
                  ?, '2026-08-21T00:00:00+00:00')
        """,
        (training_run_id, target, "a" * 64, status),
    )


def _insert_legacy_model(
    conn,
    model_id=10,
    training_run_id=1,
    target="inunda",
    selected_metric="roc_auc",
    selected_value=0.91,
):
    conn.execute(
        """
        INSERT INTO trained_models (
            model_id, training_run_id, target, algorithm,
            hyperparameters_json, preprocessing_json, feature_contract_id,
            feature_contract_sha256, ordered_features_json,
            target_transform_json, query_params_json, included_run_ids_json,
            random_seed, grouping_strategy, python_version,
            library_versions_json, selected_metric, selected_value,
            model_sha256, model_blob, created_at_utc
        ) VALUES (?, ?, ?, 'linear', '{}', '{}', 'tabular_v3_17', ?, '[]',
                  '{}', '{}', '[]', 42, 'group_kfold', '3.13', '{}', ?, ?, ?,
                  ?, '2026-08-21T01:00:00+00:00')
        """,
        (
            model_id,
            training_run_id,
            target,
            "b" * 64,
            selected_metric,
            selected_value,
            f"{model_id:064d}",
            f"model-{model_id}".encode(),
        ),
    )


def _insert_model(conn, model_id, training_run_id, target):
    conn.execute(
        """
        INSERT INTO trained_models (
            model_id, training_run_id, target, algorithm,
            hyperparameters_json, preprocessing_json, feature_contract_id,
            feature_contract_sha256, ordered_features_json,
            target_transform_json, query_params_json, included_run_ids_json,
            random_seed, grouping_strategy, python_version,
            library_versions_json, model_sha256, model_blob, created_at_utc
        ) VALUES (?, ?, ?, 'linear', '{}', '{}', 'tabular_v3_17', ?, '[]',
                  '{}', '{}', '[]', 42, 'group_kfold', '3.13', '{}', ?, ?,
                  '2026-08-21T01:00:00+00:00')
        """,
        (
            model_id,
            training_run_id,
            target,
            "b" * 64,
            f"{model_id:064d}",
            f"model-{model_id}".encode(),
        ),
    )


def _insert_promotion(
    conn,
    promotion_id,
    training_run_id,
    target,
    classifier_model_id=None,
    regressor_model_id=None,
):
    conn.execute(
        """
        INSERT INTO model_promotions (
            promotion_id, training_run_id, target, classifier_model_id,
            regressor_model_id, primary_metric, primary_value,
            tie_breakers_json, ranking_json, promoted_at_utc
        ) VALUES (?, ?, ?, ?, ?, 'roc_auc', 0.91, '[]', '{}', ?)
        """,
        (
            promotion_id,
            training_run_id,
            target,
            classifier_model_id,
            regressor_model_id,
            f"2026-08-21T02:00:{promotion_id:02d}+00:00",
        ),
    )


def test_fresh_database_applies_four_migrations_idempotently(tmp_path):
    conn = connect_database(tmp_path / "fresh.sqlite3")
    try:
        apply_migrations(conn)
        apply_migrations(conn)

        assert [tuple(row) for row in conn.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()] == [
            (1, "v17_initial"),
            (2, "model_integrity"),
            (3, "model_integrity_guards"),
            (4, "training_run_identity"),
            (5, "provenance_integrity"),
        ]
        checksums = conn.execute(
            "SELECT checksum_sha256 FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert all(len(row[0]) == 64 for row in checksums)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_upgrade_preserves_legacy_ids_metrics_and_creates_promotion(tmp_path):
    catalog_v1 = _migration_catalog(tmp_path, include_v2=False)
    conn = connect_database(tmp_path / "upgrade.sqlite3")
    try:
        apply_migrations(conn, migration_dir=catalog_v1)
        _insert_training_run(conn)
        _insert_legacy_model(conn)
        _insert_legacy_model(
            conn,
            model_id=11,
            target="vol_inundacion_m3",
            selected_metric="rmse",
            selected_value=2.5,
        )
        conn.executemany(
            """
            INSERT INTO model_metrics (
                metric_id, owner_kind, owner_id, scope, metric_name, value,
                valid
            ) VALUES (?, ?, ?, 'overall', ?, ?, 1)
            """,
            [
                (20, "training_run", 1, "duration_seconds", 3.0),
                (21, "model", 10, "roc_auc", 0.91),
                (22, "model", 11, "rmse", 2.5),
            ],
        )
        conn.commit()
        shutil.copyfile(
            SQL_DIR / "002_model_integrity.sql",
            catalog_v1 / "002_model_integrity.sql",
        )
        shutil.copyfile(
            SQL_DIR / "003_model_integrity_guards.sql",
            catalog_v1 / "003_model_integrity_guards.sql",
        )
        shutil.copyfile(
            SQL_DIR / "004_training_run_identity.sql",
            catalog_v1 / "004_training_run_identity.sql",
        )

        apply_migrations(conn, migration_dir=catalog_v1)

        assert [tuple(row) for row in conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()] == [(1,), (2,), (3,), (4,)]
        assert [tuple(row) for row in conn.execute(
            "SELECT model_id, model_blob FROM trained_models ORDER BY model_id"
        ).fetchall()] == [(10, b"model-10"), (11, b"model-11")]
        assert [tuple(row) for row in conn.execute(
            """
            SELECT metric_id, owner_kind, owner_id
            FROM model_metrics ORDER BY metric_id
            """
        ).fetchall()] == [
            (20, "training_run", 1),
            (21, "model", 10),
            (22, "model", 11),
        ]
        assert tuple(conn.execute(
            """
            SELECT target, classifier_model_id, regressor_model_id,
                   primary_metric, primary_value
            FROM model_promotions
            WHERE target='inunda'
            """
        ).fetchone()) == ("inunda", 10, None, "roc_auc", 0.91)
        assert tuple(conn.execute(
            """
            SELECT classifier_model_id, regressor_model_id
            FROM model_promotions WHERE target='system'
            """
        ).fetchone()) == (10, 11)
        assert [tuple(row) for row in conn.execute(
            """
            SELECT target, classifier_model_id, regressor_model_id
            FROM active_model_selections ORDER BY target
            """
        ).fetchall()] == [
            ("inunda", 10, None),
            ("system", 10, 11),
            ("vol_inundacion_m3", None, 11),
        ]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_orphan_legacy_metric_aborts_entire_upgrade(tmp_path):
    catalog_v1 = _migration_catalog(tmp_path, include_v2=False)
    conn = connect_database(tmp_path / "orphan.sqlite3")
    try:
        apply_migrations(conn, migration_dir=catalog_v1)
        conn.execute(
            """
            INSERT INTO model_metrics (
                metric_id, owner_kind, owner_id, scope, metric_name, valid
            ) VALUES (99, 'model', 999, 'overall', 'roc_auc', 0)
            """
        )
        conn.commit()
        shutil.copyfile(
            SQL_DIR / "002_model_integrity.sql",
            catalog_v1 / "002_model_integrity.sql",
        )

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            apply_migrations(conn, migration_dir=catalog_v1)

        assert [tuple(row) for row in conn.execute(
            "SELECT version FROM schema_migrations"
        ).fetchall()] == [(1,)]
        assert tuple(conn.execute(
            "SELECT owner_kind, owner_id FROM model_metrics WHERE metric_id=99"
        ).fetchone()) == ("model", 999)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='model_promotions'"
        ).fetchone() is None
        assert tuple(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='idx_timeseries_run_node_step'"
        ).fetchone()) == (1,)
    finally:
        conn.close()


def test_artifacts_promotions_and_selections_enforce_integrity(tmp_path):
    conn = connect_database(tmp_path / "integrity.sqlite3")
    try:
        apply_migrations(conn)
        _insert_training_run(conn, 1)
        _insert_training_run(conn, 2)
        _insert_model(conn, 10, 1, "inunda")
        _insert_model(conn, 11, 1, "inunda")
        _insert_model(conn, 12, 1, "vol_inundacion_m3")
        _insert_model(conn, 20, 2, "inunda")

        # Multiple immutable artifacts and promotions can share run/target/OOF.
        assert conn.execute(
            "SELECT COUNT(*) FROM trained_models WHERE training_run_id=1 AND target='inunda'"
        ).fetchone()[0] == 2
        _insert_promotion(conn, 1, 1, "system", 10, 12)
        _insert_promotion(conn, 2, 1, "system", 10, 12)
        _insert_promotion(conn, 3, 1, "system", 10, 12)

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            _insert_promotion(conn, 4, 1, "inunda", 20, None)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            _insert_promotion(conn, 4, 1, "inunda", 12, None)
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            _insert_promotion(conn, 4, 1, "system", 10, None)

        conn.execute(
            """
            INSERT INTO model_selections (
                selection_id, target, promotion_id, supersedes_selection_id,
                selected_at_utc
            ) VALUES (1, 'system', 1, NULL, '2026-08-21T03:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO model_selections (
                selection_id, target, promotion_id, supersedes_selection_id,
                selected_at_utc
            ) VALUES (2, 'system', 2, 1, '2026-08-21T04:00:00+00:00')
            """
        )
        # NOTE: these promotions are never finalized (no model_promotion_finalizations
        # row), so as of migration 005 Part F, active_model_selections correctly
        # excludes them (valid_model_promotions requires finalization). The
        # selection-chain superseding logic itself is still exercised directly
        # against model_selections, which is unaffected by promotion validity.
        assert tuple(conn.execute(
            """
            SELECT selection_id, promotion_id FROM model_selections
            WHERE target='system' AND NOT EXISTS (
                SELECT 1 FROM model_selections AS successor
                WHERE successor.supersedes_selection_id = model_selections.selection_id
            )
            """
        ).fetchone()) == (2, 2)
        assert conn.execute(
            "SELECT COUNT(*) FROM active_model_selections WHERE target='system'"
        ).fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError, match="active selection"):
            conn.execute(
                """
                INSERT INTO model_selections (
                    selection_id, target, promotion_id, selected_at_utc
                ) VALUES (3, 'system', 2, '2026-08-21T05:00:00+00:00')
                """
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE trained_models SET algorithm='svm' WHERE model_id=10")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM trained_models WHERE model_id=11")
        with pytest.raises(sqlite3.IntegrityError, match="immutable|identity"):
            conn.execute(
                """
                INSERT OR REPLACE INTO trained_models
                SELECT * FROM trained_models WHERE model_id=11
                """
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM model_promotions WHERE promotion_id=2")
        with pytest.raises(sqlite3.IntegrityError, match="immutable|identity"):
            conn.execute(
                """
                INSERT OR REPLACE INTO model_promotions (
                    promotion_id, training_run_id, target,
                    classifier_model_id, regressor_model_id, primary_metric,
                    primary_value, tie_breakers_json, ranking_json,
                    promoted_at_utc
                ) SELECT promotion_id, training_run_id, target,
                         classifier_model_id, regressor_model_id, 'f1', 0.5,
                         tie_breakers_json, ranking_json, promoted_at_utc
                  FROM model_promotions WHERE promotion_id=3
                """
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE model_selections SET selected_at_utc='x' WHERE selection_id=2")
        with pytest.raises(sqlite3.IntegrityError, match="self|identity|immutable"):
            conn.execute(
                """
                INSERT OR REPLACE INTO model_selections (
                    selection_id, target, promotion_id,
                    supersedes_selection_id, selected_at_utc
                ) VALUES (2, 'system', 2, 2, '2026-08-21T06:00:00+00:00')
                """
            )
        with pytest.raises(sqlite3.IntegrityError, match="self"):
            conn.execute(
                """
                INSERT INTO model_selections (
                    selection_id, target, promotion_id,
                    supersedes_selection_id, selected_at_utc
                ) VALUES (3, 'system', 2, 3, '2026-08-21T06:00:00+00:00')
                """
            )
        # Still unfinalized: active_model_selections stays empty (see NOTE above).
        assert conn.execute(
            "SELECT COUNT(*) FROM active_model_selections WHERE target='system'"
        ).fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_model_metrics_require_exactly_one_real_owner_and_are_immutable(tmp_path):
    conn = connect_database(tmp_path / "metrics.sqlite3")
    try:
        apply_migrations(conn)
        _insert_training_run(conn, 1, "inunda")
        conn.execute(
            """
            INSERT INTO model_evaluations (
                evaluation_id, training_run_id, task, algorithm,
                hyperparameters_json, fold_id, train_run_ids_json,
                validation_run_ids_json, status, fit_seconds, predict_seconds
            ) VALUES (2, 1, 'classification', 'linear', '{}', 0, '[]', '[]',
                      'COMPLETE', 1, 1)
            """
        )
        _insert_model(conn, 3, 1, "inunda")

        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(
                "INSERT INTO model_metrics(scope, metric_name, valid) VALUES ('x','m',1)"
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(
                """
                INSERT INTO model_metrics (
                    training_run_id, evaluation_id, scope, metric_name, valid
                ) VALUES (1, 2, 'x', 'm', 1)
                """
            )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                """
                INSERT INTO model_metrics (
                    model_id, scope, metric_name, valid
                ) VALUES (999, 'x', 'm', 1)
                """
            )

        conn.executemany(
            """
            INSERT INTO model_metrics (
                training_run_id, evaluation_id, model_id, scope,
                metric_name, valid
            ) VALUES (?, ?, ?, 'overall', 'score', 1)
            """,
            [(1, None, None), (None, 2, None), (None, None, 3)],
        )
        assert [tuple(row) for row in conn.execute(
            "SELECT owner_kind, owner_id FROM model_metrics ORDER BY metric_id"
        ).fetchall()] == [
            ("training_run", 1),
            ("evaluation", 2),
            ("model", 3),
        ]
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE model_metrics SET value=9 WHERE model_id=3")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM model_metrics WHERE model_id=3")
        with pytest.raises(sqlite3.IntegrityError, match="immutable|identity"):
            conn.execute(
                """
                INSERT OR REPLACE INTO model_metrics (
                    metric_id, model_id, scope, metric_name, value, valid
                ) SELECT metric_id, model_id, scope, metric_name, 9, valid
                  FROM model_metrics WHERE model_id=3
                """
            )
        with pytest.raises(sqlite3.IntegrityError, match="model evaluations cannot be deleted"):
            conn.execute("DELETE FROM model_evaluations WHERE evaluation_id=2")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("DELETE FROM trained_models WHERE model_id=3")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM training_runs WHERE training_run_id=1")
        assert conn.execute("SELECT COUNT(*) FROM model_metrics").fetchone()[0] == 3
    finally:
        conn.close()


def test_training_run_target_and_status_guard_artifacts_and_promotions(tmp_path):
    conn = connect_database(tmp_path / "training-guards.sqlite3")
    try:
        apply_migrations(conn)
        _insert_training_run(conn, 1, "inunda", "COMPLETE")
        _insert_training_run(conn, 2, "vol_inundacion_m3", "COMPLETE")
        _insert_training_run(conn, 3, "system", "RUNNING")
        _insert_training_run(conn, 4, "inunda", "PENDING")
        _insert_training_run(conn, 5, "inunda", "FAILED")

        with pytest.raises(sqlite3.IntegrityError, match="target"):
            _insert_model(conn, 10, 1, "vol_inundacion_m3")
        with pytest.raises(sqlite3.IntegrityError, match="target"):
            _insert_model(conn, 11, 2, "inunda")
        with pytest.raises(sqlite3.IntegrityError, match="RUNNING|COMPLETE"):
            _insert_model(conn, 12, 4, "inunda")
        with pytest.raises(sqlite3.IntegrityError, match="RUNNING|COMPLETE"):
            _insert_model(conn, 13, 5, "inunda")

        _insert_model(conn, 20, 3, "inunda")
        _insert_model(conn, 21, 3, "vol_inundacion_m3")
        with pytest.raises(sqlite3.IntegrityError, match="identity|immutable"):
            conn.execute(
                """
                INSERT OR REPLACE INTO training_runs
                SELECT * FROM training_runs WHERE training_run_id=3
                """
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM trained_models WHERE training_run_id=3"
        ).fetchone()[0] == 2
        with pytest.raises(sqlite3.IntegrityError, match="COMPLETE"):
            _insert_promotion(conn, 20, 3, "system", 20, 21)
        for training_run_id in (4, 5):
            with pytest.raises(sqlite3.IntegrityError, match="COMPLETE"):
                _insert_promotion(
                    conn,
                    20 + training_run_id,
                    training_run_id,
                    "inunda",
                    999,
                    None,
                )

        with pytest.raises(sqlite3.IntegrityError, match="target"):
            _insert_promotion(conn, 30, 1, "system", 20, 21)
        with pytest.raises(sqlite3.IntegrityError, match="target"):
            _insert_promotion(conn, 31, 1, "vol_inundacion_m3", None, 21)

        with pytest.raises(
            sqlite3.IntegrityError, match="training run fitting configuration is immutable"
        ):
            conn.execute("UPDATE training_runs SET target='inunda' WHERE training_run_id=3")
        with pytest.raises(sqlite3.IntegrityError, match="status"):
            conn.execute("UPDATE training_runs SET status='FAILED' WHERE training_run_id=3")

        # Plan C final transaction order: artifact while RUNNING, COMPLETE, promote.
        conn.execute(
            "UPDATE training_runs SET status='COMPLETE' WHERE training_run_id=3"
        )
        _insert_promotion(conn, 40, 3, "system", 20, 21)
        conn.execute(
            """
            INSERT INTO model_selections (
                selection_id, target, promotion_id, selected_at_utc
            ) VALUES (40, 'system', 40, '2026-08-21T08:00:00+00:00')
            """
        )
        with pytest.raises(
            sqlite3.IntegrityError, match="invalid training run status transition"
        ):
            conn.execute("UPDATE training_runs SET status='RUNNING' WHERE training_run_id=3")
        # NOTE: promotion 40 is never finalized (no model_promotion_finalizations
        # row), so as of migration 005 Part F, active_model_selections correctly
        # excludes it. The selection-chain leaf logic is still exercised directly
        # against model_selections, which is unaffected by promotion validity.
        assert conn.execute(
            """
            SELECT COUNT(*) FROM model_selections
            WHERE target='system' AND NOT EXISTS (
                SELECT 1 FROM model_selections AS successor
                WHERE successor.supersedes_selection_id = model_selections.selection_id
            )
            """
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM active_model_selections WHERE target='system'"
        ).fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_incoherent_v2_data_aborts_v3_and_rolls_back(tmp_path):
    catalog_v2 = _migration_catalog(tmp_path, include_v2=True)
    conn = connect_database(tmp_path / "incoherent-v2.sqlite3")
    try:
        apply_migrations(conn, migration_dir=catalog_v2)
        _insert_training_run(conn, 1, "inunda", "PENDING")
        _insert_model(conn, 10, 1, "vol_inundacion_m3")
        conn.commit()
        shutil.copyfile(
            SQL_DIR / "003_model_integrity_guards.sql",
            catalog_v2 / "003_model_integrity_guards.sql",
        )

        with pytest.raises(sqlite3.IntegrityError, match="003|integrity"):
            apply_migrations(conn, migration_dir=catalog_v2)

        assert [tuple(row) for row in conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()] == [(1,), (2,)]
        assert tuple(conn.execute(
            "SELECT training_run_id, target FROM trained_models WHERE model_id=10"
        ).fetchone()) == (1, "vol_inundacion_m3")
        assert conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='trigger' AND name='trained_models_immutable_delete'
            """
        ).fetchone() is None
    finally:
        conn.close()


def test_training_run_primary_key_is_immutable_with_or_without_children(tmp_path):
    conn = connect_database(tmp_path / "training-id.sqlite3")
    try:
        apply_migrations(conn)
        _insert_training_run(conn, 1, "inunda", "PENDING")
        _insert_training_run(conn, 2, "system", "RUNNING")
        _insert_model(conn, 20, 2, "inunda")

        with pytest.raises(sqlite3.IntegrityError, match="identity is immutable"):
            conn.execute(
                "UPDATE training_runs SET training_run_id=101 WHERE training_run_id=1"
            )
        with pytest.raises(sqlite3.IntegrityError, match="identity is immutable"):
            conn.execute(
                "UPDATE training_runs SET training_run_id=202 WHERE training_run_id=2"
            )

        assert [row[0] for row in conn.execute(
            "SELECT training_run_id FROM training_runs ORDER BY training_run_id"
        )] == [1, 2]
        assert conn.execute(
            "SELECT training_run_id FROM trained_models WHERE model_id=20"
        ).fetchone()[0] == 2
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
