# tests/database/test_migration_005_membership.py
import json
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


def _seed_network_and_runs(conn, n=2):
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
    for run_id in range(1, n + 1):
        conn.execute(
            """
            INSERT INTO runs (run_id, network_id, scenario_id, status,
                               config_sha256, node_count)
            VALUES (?, 1, 1, 'COMPLETE', ?, 1)
            """,
            (run_id, "d" * 64),
        )


def _insert_training_run(conn, training_run_id, included_run_ids, status="PENDING"):
    import json

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id,
            feature_contract_sha256, query_sql, query_params_json,
            included_run_ids_json, grouping_strategy, fold_count, random_seed,
            primary_metric, tie_breakers_json, python_version,
            library_versions_json, status
        ) VALUES (?, 'system', 'tabular_v3_17', ?, 'SELECT 1', '{}', ?,
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', ?)
        """,
        (training_run_id, "a" * 64, json.dumps(sorted(included_run_ids)), status),
    )


def _insert_model_evaluation(
    conn,
    evaluation_id,
    training_run_id,
    fold_id,
    train_run_ids,
    validation_run_ids,
    status="PENDING",
):
    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm,
            hyperparameters_json, fold_id, train_run_ids_json,
            validation_run_ids_json, status, fit_seconds, predict_seconds
        ) VALUES (?, ?, 'classification', 'algo', '{}', ?, ?, ?, ?, 0, 0)
        """,
        (
            evaluation_id,
            training_run_id,
            fold_id,
            json.dumps(train_run_ids),
            json.dumps(validation_run_ids),
            status,
        ),
    )


def _seed_training_run_with_full_membership(conn, training_run_id=1, fold_count=None):
    """Seeds 2 runs + a PENDING training run whose full [1, 2] membership
    is already normalized into training_run_inputs."""
    _seed_network_and_runs(conn, n=2)
    _insert_training_run(conn, training_run_id, [1, 2], status="PENDING")
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (?, 1)",
        (training_run_id,),
    )
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (?, 2)",
        (training_run_id,),
    )
    conn.commit()


def test_evaluation_running_succeeds_with_complete_disjoint_membership(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_training_run_with_full_membership(conn)
    _insert_model_evaluation(conn, 1, 1, fold_id=0, train_run_ids=[1], validation_run_ids=[2])
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'train', 1)"
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'validation', 2)"
    )
    conn.commit()

    conn.execute("UPDATE model_evaluations SET status = 'RUNNING' WHERE evaluation_id = 1")
    conn.commit()

    row = conn.execute(
        "SELECT status FROM model_evaluations WHERE evaluation_id = 1"
    ).fetchone()
    assert row[0] == "RUNNING"


def test_evaluation_running_rejected_when_train_role_is_empty(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_training_run_with_full_membership(conn)
    # train_run_ids_json is empty so the row-count-vs-JSON-count check for
    # the 'train' role is also 0=0 (satisfied); only the empty-role
    # NOT EXISTS branch can reject this.
    _insert_model_evaluation(conn, 1, 1, fold_id=0, train_run_ids=[], validation_run_ids=[2])
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'validation', 2)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("UPDATE model_evaluations SET status = 'RUNNING' WHERE evaluation_id = 1")
    conn.rollback()


def test_evaluation_running_rejected_when_validation_role_is_empty(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_training_run_with_full_membership(conn)
    # validation_run_ids_json is empty so the row-count-vs-JSON-count check
    # for the 'validation' role is also 0=0 (satisfied); only the
    # empty-role NOT EXISTS branch can reject this.
    _insert_model_evaluation(conn, 1, 1, fold_id=0, train_run_ids=[1], validation_run_ids=[])
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'train', 1)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("UPDATE model_evaluations SET status = 'RUNNING' WHERE evaluation_id = 1")
    conn.rollback()


def test_evaluation_running_rejected_when_train_and_validation_overlap(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_training_run_with_full_membership(conn)
    # run_id 1 is listed in BOTH canonical JSON arrays so the per-role
    # insert-time containment/JSON-membership guards both pass; only the
    # RUNNING-transition disjointness check should reject this.
    _insert_model_evaluation(conn, 1, 1, fold_id=0, train_run_ids=[1], validation_run_ids=[1])
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'train', 1)"
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'validation', 1)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("UPDATE model_evaluations SET status = 'RUNNING' WHERE evaluation_id = 1")
    conn.rollback()


def test_evaluation_running_rejected_when_fold_id_is_out_of_range(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    # fold_count=2 on the training run (see _insert_training_run), so
    # fold_id=2 is individually valid per model_evaluations' own
    # `CHECK(fold_id >= 0)` constraint at INSERT time, but it is out of
    # range relative to this training run's declared fold_count and must
    # be rejected specifically by the 005 RUNNING-transition trigger.
    _seed_training_run_with_full_membership(conn)
    _insert_model_evaluation(conn, 1, 1, fold_id=2, train_run_ids=[1], validation_run_ids=[2])
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'train', 1)"
    )
    conn.execute(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (1, 'validation', 2)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("UPDATE model_evaluations SET status = 'RUNNING' WHERE evaluation_id = 1")
    conn.rollback()


def test_membership_insert_requires_pending_owner(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_network_and_runs(conn)
    _insert_training_run(conn, 1, [1], status="RUNNING")
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
        )
    conn.rollback()


def test_membership_insert_must_be_in_canonical_json(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_network_and_runs(conn)
    _insert_training_run(conn, 1, [1], status="PENDING")
    conn.commit()

    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 2)"
        )
    conn.rollback()

    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
    )
    conn.commit()


def test_membership_rows_are_immutable(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_network_and_runs(conn)
    _insert_training_run(conn, 1, [1], status="PENDING")
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("DELETE FROM training_run_inputs WHERE training_run_id = 1")
    conn.rollback()


def test_running_requires_complete_and_equal_membership(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _seed_network_and_runs(conn, n=2)
    _insert_training_run(conn, 1, [1, 2], status="PENDING")
    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)"
    )
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("UPDATE training_runs SET status = 'RUNNING' WHERE training_run_id = 1")
    conn.rollback()

    conn.execute(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 2)"
    )
    conn.execute("UPDATE training_runs SET status = 'RUNNING' WHERE training_run_id = 1")
    conn.commit()


def test_running_is_rejected_when_membership_is_completely_empty(tmp_path):
    catalog = _catalog_through_005(tmp_path)
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn, migration_dir=catalog)
    _insert_training_run(conn, 1, [], status="PENDING")
    conn.commit()

    with pytest.raises(Exception):
        conn.execute("UPDATE training_runs SET status = 'RUNNING' WHERE training_run_id = 1")
    conn.rollback()
