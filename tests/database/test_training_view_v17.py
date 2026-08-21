import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_float_dtype, is_integer_dtype

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.training_queries import (
    export_training_samples_csv,
    load_training_samples,
)
from swmm_resilience.ml.contracts import (
    FEATURE_COLUMNS_V17,
    NULLABLE_FEATURE_COLUMNS_V17,
    FeatureContractError,
)


IDENTITY_COLUMNS = [
    "run_id",
    "network_id",
    "scenario_id",
    "scenario_key",
    "scenario_kind",
    "factor_mult",
    "shape_id",
    "node_id",
]
TARGET_COLUMNS = ["inunda", "vol_inundacion_m3"]


@pytest.fixture
def migrated_conn(tmp_path):
    conn = connect_database(tmp_path / "training.sqlite3")
    apply_migrations(conn)
    try:
        yield conn
    finally:
        conn.close()


def _insert_network(conn, network_id=1):
    conn.execute(
        """
        INSERT INTO networks (
            network_id, network_sha256, name, source_filename, inp_bytes,
            flow_units, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            network_id,
            f"{network_id:064x}",
            f"network-{network_id}",
            f"network-{network_id}.inp",
            b"inp",
            "LPS",
            "2026-08-21T00:00:00+00:00",
        ),
    )


def _insert_node(conn, node_pk, node_id, network_id=1):
    conn.execute(
        """
        INSERT INTO nodes (
            node_pk, network_id, node_id, node_type,
            invert_elevation_m, max_depth_m, base_inflow_lps
        ) VALUES (?, ?, ?, 'junction', ?, ?, ?)
        """,
        (node_pk, network_id, node_id, 100.0 + node_pk, 2.0, 0.5),
    )


def _insert_scenario_and_run(
    conn,
    *,
    run_id,
    status="COMPLETE",
    network_id=1,
    scenario_id=None,
    node_count=1,
):
    scenario_id = scenario_id if scenario_id is not None else 100 + run_id
    conn.execute(
        """
        INSERT INTO scenarios (
            scenario_id, network_id, scenario_key, scenario_kind,
            factor_mult, shape_id, duracion_horas, tiempo_al_pico_h,
            config_json, config_sha256
        ) VALUES (?, ?, ?, 'factor', ?, ?, ?, ?, '{}', ?)
        """,
        (
            scenario_id,
            network_id,
            f"scenario-{run_id}",
            1.25,
            f"shape-{run_id}",
            2.0,
            0.75,
            f"{scenario_id:064x}",
        ),
    )
    conn.execute(
        """
        INSERT INTO runs (
            run_id, scenario_id, network_id, status, config_sha256,
            node_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            scenario_id,
            network_id,
            status,
            f"{run_id:064x}",
            node_count,
        ),
    )


def _insert_feature(conn, run_id, node_pk, *, network_id=1, elev_fondo=None):
    base = float(run_id + node_pk)
    conn.execute(
        """
        INSERT INTO node_features (
            run_id, network_id, node_pk,
            elev_fondo, prof_max, n_tuberias_in, n_tuberias_out,
            diam_max_in, diam_max_out, pendiente_max_in, pendiente_out,
            base_inflow_lps, dist_outfall_m, n_nodos_aguas_arriba,
            q_pico_acum_base, upstream_capacity_lps, q_pico_nodo,
            q_pico_acum_escalado, duracion_horas, tiempo_al_pico_h,
            feature_contract_id
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            run_id,
            network_id,
            node_pk,
            base if elev_fondo is None else elev_fondo,
            2.0,
            1,
            1,
            0.8,
            0.7,
            0.01,
            0.02,
            0.5,
            125.0,
            3,
            12.0,
            18.0,
            4.0,
            8.0,
            2.0,
            0.75,
            "tabular_v3_17",
        ),
    )


def _insert_result(conn, run_id, node_pk, *, network_id=1, inunda=0):
    conn.execute(
        """
        INSERT INTO node_results (
            run_id, network_id, node_pk, inunda, vol_inundacion_m3
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, network_id, node_pk, inunda, 10.0 + node_pk),
    )


def _target_validation_connection(*, inunda=1, vol_inundacion_m3=12.5):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            node_count INTEGER
        );
        CREATE TABLE node_features (run_id INTEGER, node_pk INTEGER);
        CREATE TABLE node_results (run_id INTEGER, node_pk INTEGER);
        INSERT INTO runs VALUES (1, 'COMPLETE', 1);
        INSERT INTO node_features VALUES (1, 10);
        INSERT INTO node_results VALUES (1, 10);
        """
    )
    row = {
        "run_id": 1,
        "network_id": 1,
        "scenario_id": 1,
        "scenario_key": "scenario-1",
        "scenario_kind": "factor",
        "factor_mult": 1.0,
        "shape_id": "shape-1",
        "node_id": "node-10",
        **{
            column: float(index + 1)
            for index, column in enumerate(FEATURE_COLUMNS_V17)
        },
        "inunda": inunda,
        "vol_inundacion_m3": vol_inundacion_m3,
    }
    pd.DataFrame([row]).to_sql(
        "training_samples_v17",
        conn,
        index=False,
    )
    return conn


def test_training_view_is_flat_canonical_and_deterministic(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 20, "B-node")
    _insert_node(migrated_conn, 10, "A-node")
    _insert_scenario_and_run(migrated_conn, run_id=1, node_count=2)
    _insert_feature(migrated_conn, 1, 20)
    _insert_feature(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 20, inunda=1)
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    frame = load_training_samples(migrated_conn)

    assert frame.columns.tolist() == (
        IDENTITY_COLUMNS + list(FEATURE_COLUMNS_V17) + TARGET_COLUMNS
    )
    assert frame[["run_id", "node_id"]].values.tolist() == [
        [1, "A-node"],
        [1, "B-node"],
    ]
    assert frame["inunda"].tolist() == [0, 1]


def test_real_view_and_loader_exclude_non_complete_runs(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    for run_id, status in [(1, "COMPLETE"), (2, "FAILED"), (3, "RUNNING")]:
        _insert_scenario_and_run(
            migrated_conn,
            run_id=run_id,
            status=status,
        )
        _insert_feature(migrated_conn, run_id, 10)
        if status == "COMPLETE":
            _insert_result(migrated_conn, run_id, 10)
    migrated_conn.commit()

    view_rows = migrated_conn.execute(
        "SELECT run_id, node_id FROM training_samples_v17 ORDER BY run_id"
    ).fetchall()
    frame = load_training_samples(migrated_conn)

    assert [tuple(row) for row in view_rows] == [(1, "node-10")]
    assert frame["run_id"].tolist() == [1]


def test_loader_rejects_non_integer_run_ids_before_query(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    for run_id in (1, 2):
        _insert_scenario_and_run(migrated_conn, run_id=run_id)
        _insert_feature(migrated_conn, run_id, 10)
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    with pytest.raises(ValueError, match="positive Python int"):
        load_training_samples(
            migrated_conn,
            run_ids=[1, "2) OR 1=1 --"],
        )


def test_loader_rejects_empty_run_id_selection(migrated_conn):
    with pytest.raises(ValueError, match="run_ids cannot be empty"):
        load_training_samples(migrated_conn, run_ids=[])


@pytest.mark.parametrize(
    "invalid_run_id",
    [True, "1", 1.0, 0, -1, -(2**63) - 1],
)
def test_loader_rejects_invalid_explicit_run_ids(
    migrated_conn,
    invalid_run_id,
):
    with pytest.raises(ValueError, match="positive Python int"):
        load_training_samples(migrated_conn, run_ids=[invalid_run_id])


def test_loader_rejects_run_id_above_sqlite_integer_range(migrated_conn):
    with pytest.raises(ValueError, match="SQLite INTEGER range"):
        load_training_samples(migrated_conn, run_ids=[2**63])


def test_loader_accepts_sqlite_max_integer_before_database_selection(
    migrated_conn,
):
    with pytest.raises(ValueError, match="missing run_ids"):
        load_training_samples(migrated_conn, run_ids=[2**63 - 1])


@pytest.mark.parametrize(
    ("requested", "expected_detail"),
    [
        ([999], "missing run_ids: [999]"),
        ([2], "non-COMPLETE run_ids: [2]"),
        ([1, 999], "missing run_ids: [999]"),
        ([1, 2], "non-COMPLETE run_ids: [2]"),
    ],
)
def test_loader_rejects_missing_or_non_complete_explicit_run_ids(
    migrated_conn,
    requested,
    expected_detail,
):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 10)
    _insert_scenario_and_run(migrated_conn, run_id=2, status="FAILED")
    migrated_conn.commit()

    with pytest.raises(ValueError) as error:
        load_training_samples(migrated_conn, run_ids=requested)

    assert expected_detail in str(error.value)


def test_loader_deduplicates_valid_explicit_run_ids(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    frame = load_training_samples(migrated_conn, run_ids=[1, 1, 1])

    assert frame[["run_id", "node_id"]].values.tolist() == [[1, "node-10"]]


def test_loader_rejects_absence_of_complete_samples(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1, status="FAILED")
    _insert_feature(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    with pytest.raises(ValueError, match="No COMPLETE v17 training samples found"):
        load_training_samples(migrated_conn)


def test_loader_rejects_missing_result_keys(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_node(migrated_conn, 20, "node-20")
    _insert_scenario_and_run(migrated_conn, run_id=1, node_count=2)
    _insert_feature(migrated_conn, 1, 10)
    _insert_feature(migrated_conn, 1, 20)
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    with pytest.raises(ValueError, match="feature/result row count"):
        load_training_samples(migrated_conn)


def test_loader_rejects_equal_counts_with_different_node_keys(migrated_conn):
    _insert_network(migrated_conn)
    for node_pk in (10, 20, 30):
        _insert_node(migrated_conn, node_pk, f"node-{node_pk}")
    _insert_scenario_and_run(migrated_conn, run_id=1, node_count=2)
    _insert_feature(migrated_conn, 1, 10)
    _insert_feature(migrated_conn, 1, 20)
    _insert_result(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 30)
    migrated_conn.commit()

    with pytest.raises(ValueError, match="feature/result row count") as error:
        load_training_samples(migrated_conn)

    assert "missing result (1, 20)" in str(error.value)
    assert "missing feature (1, 30)" in str(error.value)


def test_loader_rejects_symmetric_child_deletion_against_node_count(
    migrated_conn,
):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_node(migrated_conn, 20, "node-20")
    _insert_scenario_and_run(migrated_conn, run_id=1, node_count=2)
    for node_pk in (10, 20):
        _insert_feature(migrated_conn, 1, node_pk)
        _insert_result(migrated_conn, 1, node_pk)
    migrated_conn.execute(
        "DELETE FROM node_features WHERE run_id = 1 AND node_pk = 20"
    )
    migrated_conn.execute(
        "DELETE FROM node_results WHERE run_id = 1 AND node_pk = 20"
    )
    migrated_conn.commit()

    with pytest.raises(ValueError, match="cardinality mismatch") as error:
        load_training_samples(migrated_conn)

    assert "node_count=2, feature_count=1, result_count=1" in str(error.value)


def test_loader_rejects_complete_run_with_no_child_rows(migrated_conn):
    _insert_network(migrated_conn)
    _insert_scenario_and_run(migrated_conn, run_id=1, node_count=1)
    migrated_conn.commit()

    with pytest.raises(ValueError, match="cardinality mismatch") as error:
        load_training_samples(migrated_conn)

    assert "node_count=1, feature_count=0, result_count=0" in str(error.value)


@pytest.mark.parametrize("node_count", [None, 0, -1, 1.5])
def test_loader_rejects_invalid_persisted_node_count(
    migrated_conn,
    node_count,
):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(
        migrated_conn,
        run_id=1,
        node_count=node_count,
    )
    _insert_feature(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    with pytest.raises(ValueError, match="positive integer") as error:
        load_training_samples(migrated_conn)

    assert "run 1" in str(error.value)


def _matches_snapshot_boundary(statement, boundary):
    sql = " ".join(statement.lower().split())
    if boundary == "selection":
        return (
            "select run_id, status" in sql
            and "from runs" in sql
            and "where status = 'complete'" in sql
        )
    if boundary == "cardinality":
        return (
            "run.node_count" in sql
            and "as feature_count" in sql
            and "as result_count" in sql
        )
    if boundary == "key_symmetry":
        return "with selected_runs as" in sql and "mismatches as" in sql
    if boundary == "final_view":
        return "from training_samples_v17" in sql
    raise AssertionError(f"Unknown snapshot boundary: {boundary}")


CRITICAL_SNAPSHOT_READS = (
    "selection",
    "cardinality",
    "key_symmetry",
    "final_view",
)


@pytest.mark.parametrize(
    "mutation_boundary",
    ["cardinality", "key_symmetry", "final_view"],
)
def test_loader_reads_every_validation_boundary_from_one_wal_snapshot(
    tmp_path,
    mutation_boundary,
):
    database_path = tmp_path / "snapshot.sqlite3"
    reader = connect_database(database_path)
    writer = None
    mutated = False
    mutation_inside_loader_savepoint = False
    loader_savepoint_open = False
    critical_read_states = {}
    try:
        apply_migrations(reader)
        _insert_network(reader)
        _insert_node(reader, 10, "node-10")
        _insert_node(reader, 20, "node-20")
        _insert_scenario_and_run(reader, run_id=1, node_count=2)
        _insert_feature(reader, 1, 10)
        _insert_feature(reader, 1, 20)
        _insert_result(reader, 1, 10)
        _insert_result(reader, 1, 20)
        reader.commit()
        writer = connect_database(database_path)

        def delete_symmetric_rows_at_boundary(statement):
            nonlocal loader_savepoint_open
            nonlocal mutated
            nonlocal mutation_inside_loader_savepoint
            sql = " ".join(statement.lower().split())
            if sql.startswith("savepoint training_samples_"):
                loader_savepoint_open = True
                return
            if sql.startswith("release savepoint training_samples_"):
                loader_savepoint_open = False
                return
            for critical_read in CRITICAL_SNAPSHOT_READS:
                if _matches_snapshot_boundary(statement, critical_read):
                    critical_read_states.setdefault(critical_read, []).append(
                        loader_savepoint_open
                    )
            if mutated or not _matches_snapshot_boundary(
                statement,
                mutation_boundary,
            ):
                return
            mutated = True
            mutation_inside_loader_savepoint = loader_savepoint_open
            writer.execute(
                "DELETE FROM node_features WHERE run_id = 1 AND node_pk = 20"
            )
            writer.execute(
                "DELETE FROM node_results WHERE run_id = 1 AND node_pk = 20"
            )
            writer.commit()

        reader.set_trace_callback(delete_symmetric_rows_at_boundary)
        reader.execute("BEGIN")
        frame = load_training_samples(reader)
        reader.set_trace_callback(None)

        assert mutated
        assert mutation_inside_loader_savepoint
        assert set(critical_read_states) == set(CRITICAL_SNAPSHOT_READS)
        assert all(
            all(states) for states in critical_read_states.values()
        )
        assert frame["node_id"].tolist() == ["node-10", "node-20"]
        assert reader.in_transaction
        assert reader.execute(
            "SELECT COUNT(*) FROM node_features WHERE run_id = 1"
        ).fetchone()[0] == 2
        assert reader.execute(
            "SELECT COUNT(*) FROM node_results WHERE run_id = 1"
        ).fetchone()[0] == 2
        assert writer.execute(
            "SELECT COUNT(*) FROM node_features WHERE run_id = 1"
        ).fetchone()[0] == 1
        assert writer.execute(
            "SELECT COUNT(*) FROM node_results WHERE run_id = 1"
        ).fetchone()[0] == 1
        reader.rollback()
        assert not reader.in_transaction
        assert reader.execute(
            "SELECT COUNT(*) FROM node_features WHERE run_id = 1"
        ).fetchone()[0] == 1
        assert reader.execute(
            "SELECT COUNT(*) FROM node_results WHERE run_id = 1"
        ).fetchone()[0] == 1
    finally:
        reader.set_trace_callback(None)
        if reader.in_transaction:
            reader.rollback()
        if writer is not None:
            writer.close()
        reader.close()


def test_loader_snapshot_preserves_callers_transaction_on_success(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()
    migrated_conn.execute(
        "UPDATE networks SET name = 'caller-pending' WHERE network_id = 1"
    )

    frame = load_training_samples(migrated_conn)

    assert frame["node_id"].tolist() == ["node-10"]
    assert migrated_conn.in_transaction
    assert migrated_conn.execute(
        "SELECT name FROM networks WHERE network_id = 1"
    ).fetchone()[0] == "caller-pending"
    migrated_conn.rollback()
    assert migrated_conn.execute(
        "SELECT name FROM networks WHERE network_id = 1"
    ).fetchone()[0] == "network-1"


def test_loader_snapshot_preserves_callers_transaction_on_failure(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10)
    migrated_conn.commit()
    migrated_conn.execute(
        "UPDATE networks SET name = 'caller-pending' WHERE network_id = 1"
    )

    with pytest.raises(ValueError, match="feature/result row count"):
        load_training_samples(migrated_conn)

    assert migrated_conn.in_transaction
    assert migrated_conn.execute(
        "SELECT name FROM networks WHERE network_id = 1"
    ).fetchone()[0] == "caller-pending"
    migrated_conn.rollback()
    assert migrated_conn.execute(
        "SELECT name FROM networks WHERE network_id = 1"
    ).fetchone()[0] == "network-1"


def test_loader_validates_feature_contract_before_returning(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10, elev_fondo="not-numeric")
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    with pytest.raises(FeatureContractError, match="elev_fondo"):
        load_training_samples(migrated_conn)


def test_loader_returns_nullable_features_as_normalized_floats(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10)
    migrated_conn.execute(
        """
        UPDATE node_features
        SET diam_max_in = NULL,
            diam_max_out = NULL,
            pendiente_max_in = NULL,
            pendiente_out = NULL,
            dist_outfall_m = NULL,
            upstream_capacity_lps = NULL
        """
    )
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    frame = load_training_samples(migrated_conn)

    for column in NULLABLE_FEATURE_COLUMNS_V17:
        assert frame[column].isna().all()
        assert is_float_dtype(frame[column].dtype), (
            column,
            frame[column].dtype,
        )


def test_loader_normalizes_valid_targets_to_canonical_dtypes():
    conn = _target_validation_connection(
        inunda="1",
        vol_inundacion_m3="12.5",
    )
    try:
        frame = load_training_samples(conn)
    finally:
        conn.close()

    assert frame["inunda"].tolist() == [1]
    assert is_integer_dtype(frame["inunda"].dtype)
    assert frame["vol_inundacion_m3"].tolist() == [12.5]
    assert is_float_dtype(frame["vol_inundacion_m3"].dtype)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "inunda must not contain nulls"),
        (float("nan"), "inunda must not contain nulls"),
        (float("inf"), "inunda must contain only finite values"),
        (float("-inf"), "inunda must contain only finite values"),
        ("not-a-number", "inunda must be numeric"),
        (-1, "inunda must contain only 0 or 1"),
        (2, "inunda must contain only 0 or 1"),
        (0.5, "inunda must contain only 0 or 1"),
    ],
)
def test_loader_rejects_invalid_classification_targets(value, message):
    conn = _target_validation_connection(inunda=value)
    try:
        with pytest.raises(ValueError, match=message):
            load_training_samples(conn)
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "vol_inundacion_m3 must not contain nulls"),
        (float("nan"), "vol_inundacion_m3 must not contain nulls"),
        (float("inf"), "vol_inundacion_m3 must contain only finite values"),
        (float("-inf"), "vol_inundacion_m3 must contain only finite values"),
        ("not-a-number", "vol_inundacion_m3 must be numeric"),
        (-0.01, "vol_inundacion_m3 must be non-negative"),
    ],
)
def test_loader_rejects_invalid_regression_targets(value, message):
    conn = _target_validation_connection(vol_inundacion_m3=value)
    try:
        with pytest.raises(ValueError, match=message):
            load_training_samples(conn)
    finally:
        conn.close()


def test_csv_exists_only_after_explicit_export(migrated_conn, tmp_path):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 10, inunda=1)
    migrated_conn.commit()
    output = tmp_path / "nested" / "training.csv"

    frame = load_training_samples(migrated_conn)
    assert not output.exists()

    result = export_training_samples_csv(
        migrated_conn,
        output,
        run_ids=[1],
    )

    assert result == output
    assert output.exists()
    exported = pd.read_csv(output)
    assert exported.columns.tolist() == frame.columns.tolist()
    assert exported[["run_id", "node_id"]].values.tolist() == [[1, "node-10"]]


def test_export_propagates_incomplete_data_error_without_creating_csv(
    migrated_conn,
    tmp_path,
):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10)
    migrated_conn.commit()
    output = tmp_path / "must-not-exist.csv"

    with pytest.raises(ValueError, match="feature/result row count"):
        export_training_samples_csv(migrated_conn, output)

    assert not output.exists()
    assert not migrated_conn.in_transaction


def test_export_write_failure_preserves_existing_csv_and_removes_temporary(
    migrated_conn,
    tmp_path,
    monkeypatch,
):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()
    output = tmp_path / "training.csv"
    original = b"existing,content\n1,kept\n"
    output.write_bytes(original)

    def write_partial_then_fail(_frame, destination, *, index):
        assert index is False
        Path(destination).write_text("partial", encoding="utf-8")
        raise OSError("simulated CSV write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", write_partial_then_fail)

    with pytest.raises(OSError, match="simulated CSV write failure"):
        export_training_samples_csv(migrated_conn, output)

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".training.csv.*.tmp")) == []
