import pandas as pd
import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.training_queries import (
    export_training_samples_csv,
    load_training_samples,
)
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17, FeatureContractError


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
            run_id, scenario_id, network_id, status, config_sha256
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, scenario_id, network_id, status, f"{run_id:064x}"),
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


def test_training_view_is_flat_canonical_and_deterministic(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 20, "B-node")
    _insert_node(migrated_conn, 10, "A-node")
    _insert_scenario_and_run(migrated_conn, run_id=1)
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


def test_loader_filters_run_ids_with_bound_parameters(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    for run_id in (1, 2):
        _insert_scenario_and_run(migrated_conn, run_id=run_id)
        _insert_feature(migrated_conn, run_id, 10)
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    frame = load_training_samples(
        migrated_conn,
        run_ids=[1, "2) OR 1=1 --"],
    )

    assert frame["run_id"].tolist() == [1]


def test_loader_rejects_empty_run_id_selection(migrated_conn):
    with pytest.raises(ValueError, match="run_ids cannot be empty"):
        load_training_samples(migrated_conn, run_ids=[])


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
    _insert_scenario_and_run(migrated_conn, run_id=1)
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
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10)
    _insert_feature(migrated_conn, 1, 20)
    _insert_result(migrated_conn, 1, 10)
    _insert_result(migrated_conn, 1, 30)
    migrated_conn.commit()

    with pytest.raises(ValueError, match="feature/result row count") as error:
        load_training_samples(migrated_conn)

    assert "missing result (1, 20)" in str(error.value)
    assert "missing feature (1, 30)" in str(error.value)


def test_loader_validates_feature_contract_before_returning(migrated_conn):
    _insert_network(migrated_conn)
    _insert_node(migrated_conn, 10, "node-10")
    _insert_scenario_and_run(migrated_conn, run_id=1)
    _insert_feature(migrated_conn, 1, 10, elev_fondo="not-numeric")
    _insert_result(migrated_conn, 1, 10)
    migrated_conn.commit()

    with pytest.raises(FeatureContractError, match="elev_fondo"):
        load_training_samples(migrated_conn)


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
