from pathlib import Path
import hashlib
import json

import pandas as pd
import pytest

from swmm_resilience.config import load_config
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.csv_backfill import (
    backfill_networks_and_runs,
    persist_training_run,
)
from swmm_resilience.database.hydraulic_quality import build_hydraulic_quality_decision
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _synthetic_dataset(n_nodes=8, factors=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), shapes=("base", "storm_a")):
    """Small dataset matching dataset_final.csv's real column set, with a
    genuine flooded/not-flooded split so classifier CV has both classes."""
    rows = []
    for shape_id in shapes:
        for factor in factors:
            for node_idx in range(n_nodes):
                node_id = f"N{node_idx}"
                flooded = 1 if (node_idx < 3 and factor >= 1.5) else 0
                rows.append({
                    "node_id": node_id,
                    "elev_fondo": 100.0 + node_idx,
                    "prof_max": 1.5,
                    "n_tuberias_in": 1,
                    "n_tuberias_out": 1,
                    "diam_max_in": 0.2,
                    "diam_max_out": 0.2,
                    "pendiente_max_in": 0.01,
                    "pendiente_out": 0.01,
                    "base_inflow_lps": 5.0 + node_idx,
                    "dist_outfall_m": 100.0 + node_idx * 10,
                    "n_nodos_aguas_arriba": node_idx,
                    "q_pico_acum_base": 10.0,
                    "upstream_capacity_lps": 50.0,
                    "coord_x": float(node_idx),
                    "coord_y": float(node_idx * 2),
                    "factor_mult": factor,
                    "q_pico_nodo": 2.0 * factor,
                    "q_pico_acum_escalado": 10.0 * factor,
                    "duracion_horas": 3.0,
                    "tiempo_al_pico_h": 0.7,
                    "shape_id": shape_id,
                    "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                    "inunda": flooded,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def migrated_managed_conn(tmp_path):
    conn = connect_managed_database(tmp_path / "backfill.sqlite3")
    apply_migrations(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def fake_inp(tmp_path):
    path = tmp_path / "network.inp"
    path.write_text("[TITLE]\ntest network\n")
    return path


def test_backfill_creates_expected_row_counts(migrated_managed_conn, fake_inp):
    df = _synthetic_dataset()
    info = backfill_networks_and_runs(migrated_managed_conn, df, fake_inp, "Test Network")

    n_nodes = df["node_id"].nunique()
    n_scenarios = df[["shape_id", "factor_mult"]].drop_duplicates().shape[0]

    assert len(info["node_pk_by_id"]) == n_nodes
    assert len(info["run_id_by_key"]) == n_scenarios
    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM networks").fetchone()[0] == 1
    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == n_nodes
    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0] == n_scenarios
    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == n_scenarios
    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM node_features").fetchone()[0] == len(df)
    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] == len(df)
    assert migrated_managed_conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert [tuple(r) for r in migrated_managed_conn.execute(
        "SELECT DISTINCT status FROM runs"
    ).fetchall()] == [("COMPLETE",)]


def test_backfill_persists_only_explicitly_eligible_hydraulic_quality(
    migrated_managed_conn, fake_inp,
):
    dataset = _synthetic_dataset(n_nodes=2, factors=(1.0,), shapes=("base",))
    dataset["continuity_error_pct"] = 0.5
    decision = build_hydraulic_quality_decision(
        continuity_error_pct=0.5,
        max_continuity_error_pct=5.0,
        continuity_limit_rationale="Provisional fixture limit.",
        finite_coverage_ok=True,
        drainage_state="drained",
        initial_condition={"policy_id": "fixture", "state_sha256": "a" * 64},
        flooding_report_manifest={
            "parser": "fixture", "parser_version": "1", "report_sha256": "b" * 64,
        },
    )
    info = backfill_networks_and_runs(
        migrated_managed_conn,
        dataset,
        fake_inp,
        "Test Network",
        hydraulic_quality_decisions={("base", 1.0): decision},
    )

    run_id = next(iter(info["run_id_by_key"].values()))
    row = migrated_managed_conn.execute(
        "SELECT eligible, continuity_error_pct, drainage_state "
        "FROM run_hydraulic_quality WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    assert tuple(row) == (1, 0.5, "drained")


def test_backfill_quality_mode_fails_closed_when_scenario_decision_is_missing(
    migrated_managed_conn, fake_inp,
):
    dataset = _synthetic_dataset(n_nodes=2, factors=(1.0,), shapes=("base",))
    with pytest.raises(ValueError, match="requires an eligible hydraulic-quality decision"):
        backfill_networks_and_runs(
            migrated_managed_conn,
            dataset,
            fake_inp,
            "Test Network",
            hydraulic_quality_decisions={},
        )
    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM networks").fetchone()[0] == 0


def test_backfill_rejects_duplicate_scenario_node_rows_before_writing(
    migrated_managed_conn, fake_inp
):
    dataset = _synthetic_dataset(n_nodes=2, factors=(1.0,), shapes=("base",))
    duplicated = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=r"network_id=.*shape_id='base'.*node_id='N0'"):
        backfill_networks_and_runs(
            migrated_managed_conn, duplicated, fake_inp, "Test Network"
        )

    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM networks").fetchone()[0] == 0


def test_backfill_records_network_dataset_and_event_source_hashes(
    migrated_managed_conn, fake_inp, tmp_path
):
    dataset = _synthetic_dataset(n_nodes=2, factors=(1.0,), shapes=("base", "storm_a"))
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("source csv bytes", encoding="utf-8")
    shape_dir = tmp_path / "shapes"
    shape_dir.mkdir()
    shape_path = shape_dir / "storm_a.csv"
    shape_path.write_text("time,q\\n0,0\\n1,1\\n", encoding="utf-8")

    info = backfill_networks_and_runs(
        migrated_managed_conn,
        dataset,
        fake_inp,
        "Test Network",
        source_dataset_path=csv_path,
        hydrograph_shapes_dir=shape_dir,
    )
    rows = migrated_managed_conn.execute(
        "SELECT s.shape_id, s.config_json FROM scenarios AS s "
        "WHERE s.network_id = ? ORDER BY s.shape_id",
        (info["network_id"],),
    ).fetchall()
    by_shape = {shape: json.loads(config) for shape, config in rows}
    network_id = info["network_id"]

    for shape in ("base", "storm_a"):
        manifest = by_shape[shape]["source_manifest"]
        assert manifest["network_id"] == network_id
        assert manifest["network_inp_sha256"] == _sha256(fake_inp.read_bytes())
        assert manifest["training_dataset"]["sha256"] == _sha256(csv_path.read_bytes())
        assert len(manifest["event_series_sha256"]) == 64
    assert by_shape["storm_a"]["source_manifest"]["event_series_components"][
        "hydrograph_sha256"
    ] == _sha256(shape_path.read_bytes())


def test_backfill_reuses_existing_network_on_rerun(migrated_managed_conn, fake_inp):
    df = _synthetic_dataset()
    info1 = backfill_networks_and_runs(migrated_managed_conn, df, fake_inp, "Test Network")
    # a disjoint second batch of scenarios against the SAME network/.inp
    df2 = _synthetic_dataset(factors=(4.0,), shapes=("storm_b",))
    info2 = backfill_networks_and_runs(migrated_managed_conn, df2, fake_inp, "Test Network")

    assert info1["network_id"] == info2["network_id"]
    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM networks").fetchone()[0] == 1
    # nodes are reused too (same node_ids), not duplicated
    assert migrated_managed_conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == df["node_id"].nunique()


def test_persist_training_run_writes_full_lifecycle_evidence(migrated_managed_conn, fake_inp, tmp_path):
    df = _synthetic_dataset()
    config = load_config("config.yaml")
    info = backfill_networks_and_runs(migrated_managed_conn, df, fake_inp, "Test Network")

    training_run_id = persist_training_run(
        migrated_managed_conn, df, info["run_id_by_key"], info["node_pk_by_id"], config
    )

    conn = migrated_managed_conn
    assert tuple(conn.execute(
        "SELECT status FROM training_runs WHERE training_run_id = ?", (training_run_id,)
    ).fetchone()) == ("COMPLETE",)

    # 5 folds x 2 tasks (classification, regression)
    assert conn.execute(
        "SELECT COUNT(*) FROM model_evaluations WHERE training_run_id = ?", (training_run_id,)
    ).fetchone()[0] == 10
    assert conn.execute(
        "SELECT COUNT(*) FROM model_evaluations WHERE training_run_id = ? AND status != 'COMPLETE'",
        (training_run_id,),
    ).fetchone()[0] == 0

    assert conn.execute("SELECT COUNT(*) FROM oof_predictions").fetchone()[0] > 0
    assert conn.execute(
        "SELECT COUNT(DISTINCT target) FROM trained_models WHERE training_run_id = ?",
        (training_run_id,),
    ).fetchone()[0] == 2

    metric_rows = conn.execute(
        "SELECT metric_name, value FROM model_metrics WHERE owner_kind = 'model'"
    ).fetchall()
    metric_names = {row[0] for row in metric_rows}
    assert metric_names == {"f1", "rmse"}

    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    # minimal scope: no candidates/rankings/promotions/selections built
    assert conn.execute("SELECT COUNT(*) FROM model_candidates").fetchone()[0] == 0
    assert conn.execute("SELECT * FROM active_model_selections").fetchall() == []


def test_persist_training_run_rejects_zero_flood_dataset(migrated_managed_conn, fake_inp):
    """A dataset with no flooded rows cannot train the regressor. Reject it
    up front (like trainer.train_models) instead of committing a partial
    RUNNING training run and then crashing on the final regressor fit."""
    df = _synthetic_dataset()
    df["inunda"] = 0
    df["vol_inundacion_m3"] = 0.0
    config = load_config("config.yaml")
    info = backfill_networks_and_runs(migrated_managed_conn, df, fake_inp, "Test Network")

    with pytest.raises(ValueError, match="There are no flooded rows"):
        persist_training_run(
            migrated_managed_conn, df, info["run_id_by_key"], info["node_pk_by_id"], config
        )

    assert migrated_managed_conn.execute(
        "SELECT COUNT(*) FROM training_runs"
    ).fetchone()[0] == 0
    assert migrated_managed_conn.execute(
        "SELECT COUNT(*) FROM model_evaluations"
    ).fetchone()[0] == 0


def test_persist_training_run_records_configured_algorithm(
    migrated_managed_conn, fake_inp, tiny_config_factory
):
    """The evidence chain must name the algorithm that was actually trained,
    not a hardcoded 'xgboost'."""
    df = _synthetic_dataset()
    config = tiny_config_factory(algorithm="random_forest")
    info = backfill_networks_and_runs(migrated_managed_conn, df, fake_inp, "Test Network")
    persist_training_run(
        migrated_managed_conn, df, info["run_id_by_key"], info["node_pk_by_id"], config
    )

    eval_algos = {
        r[0] for r in migrated_managed_conn.execute(
            "SELECT DISTINCT algorithm FROM model_evaluations"
        )
    }
    model_algos = {
        r[0] for r in migrated_managed_conn.execute(
            "SELECT DISTINCT algorithm FROM trained_models"
        )
    }
    assert eval_algos == {"random_forest"}
    assert model_algos == {"random_forest"}


def test_persist_training_run_uses_full_feature_contract(migrated_managed_conn, fake_inp):
    df = _synthetic_dataset()
    config = load_config("config.yaml")
    info = backfill_networks_and_runs(migrated_managed_conn, df, fake_inp, "Test Network")
    training_run_id = persist_training_run(
        migrated_managed_conn, df, info["run_id_by_key"], info["node_pk_by_id"], config
    )
    ordered = migrated_managed_conn.execute(
        "SELECT ordered_features_json FROM trained_models WHERE training_run_id = ? LIMIT 1",
        (training_run_id,),
    ).fetchone()[0]
    import json
    assert json.loads(ordered) == list(FEATURE_COLUMNS_V17)


def test_backfill_persists_continuity_error_pct_when_present(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    dataset = _synthetic_dataset()
    dataset["continuity_error_pct"] = -7.31

    conn = connect_managed_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    inp_path = tmp_path / "base.inp"
    inp_path.write_bytes(b"fake inp bytes")

    info = backfill_networks_and_runs(conn, dataset, inp_path, "test-network")

    run_id = next(iter(info["run_id_by_key"].values()))
    row = conn.execute(
        "SELECT continuity_error_pct FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row[0] == pytest.approx(-7.31)


def test_backfill_leaves_continuity_error_pct_null_when_column_absent(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    dataset = _synthetic_dataset()
    assert "continuity_error_pct" not in dataset.columns

    conn = connect_managed_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    inp_path = tmp_path / "base.inp"
    inp_path.write_bytes(b"fake inp bytes")

    info = backfill_networks_and_runs(conn, dataset, inp_path, "test-network")

    run_id = next(iter(info["run_id_by_key"].values()))
    row = conn.execute(
        "SELECT continuity_error_pct FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row[0] is None


def _synthetic_dataset_v4_21(n_nodes=8, factors=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), shapes=("base", "storm_a")):
    rows = []
    for shape_id in shapes:
        for factor in factors:
            for node_idx in range(n_nodes):
                node_id = f"N{node_idx}"
                flooded = 1 if (node_idx < 3 and factor >= 1.5) else 0
                rows.append({
                    "node_id": node_id,
                    "elev_fondo": 100.0 + node_idx,
                    "prof_max": 1.5,
                    "n_tuberias_in": 1,
                    "n_tuberias_out": 1,
                    "diam_max_in": 0.2,
                    "diam_max_out": 0.2,
                    "pendiente_max_in": 0.01,
                    "pendiente_out": 0.01,
                    "base_inflow_lps": 5.0 + node_idx,
                    "dist_outfall_m": 100.0 + node_idx * 10,
                    "n_nodos_aguas_arriba": node_idx,
                    "q_pico_acum_base": 10.0,
                    "upstream_capacity_lps": 50.0,
                    "coord_x": float(node_idx),
                    "coord_y": float(node_idx * 2),
                    "factor_mult": factor,
                    "q_pico_nodo": 2.0 * factor,
                    "q_pico_acum_escalado": 10.0 * factor,
                    "volumen_local_integrado_m3": 1.0 * factor,
                    "volumen_tributario_integrado_m3": 2.0 * factor,
                    "duracion_efectiva_tributario_h": 1.5,
                    "tiempo_primer_pico_tributario_h": 0.5,
                    "tiempo_sobre_capacidad_h": None,
                    "integral_exceso_capacidad_m3": None,
                    "shape_id": shape_id,
                    "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                    "inunda": flooded,
                })
    return pd.DataFrame(rows)


def test_backfill_tributary_features_persists_the_21_feature_contract(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations
    from swmm_resilience.database.csv_backfill import (
        backfill_networks_and_runs, backfill_tributary_features,
    )

    dataset_v17 = _synthetic_dataset()
    dataset_v21 = _synthetic_dataset_v4_21()

    conn = connect_managed_database(tmp_path / "test.sqlite3")
    apply_migrations(conn)
    inp_path = tmp_path / "base.inp"
    inp_path.write_bytes(b"fake inp bytes")

    info = backfill_networks_and_runs(conn, dataset_v17, inp_path, "test-network")
    v4_source_path = tmp_path / "dataset_v4_21.csv"
    v4_manifest_path = tmp_path / "dataset_v4_21.csv.manifest.json"
    v4_source_path.write_text("source v4 data", encoding="utf-8")
    v4_manifest_path.write_text("validated manifest bytes", encoding="utf-8")
    inserted = backfill_tributary_features(
        conn, dataset_v21, info["run_id_by_key"], info["node_pk_by_id"], info["network_id"],
        source_dataset_path=v4_source_path,
        source_manifest_path=v4_manifest_path,
    )

    assert inserted == len(dataset_v21)
    row = conn.execute(
        "SELECT volumen_local_integrado_m3, feature_contract_id FROM node_features_v4_21 LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[1] == "tabular_v4_21"
    manifest = conn.execute(
        "SELECT config_json FROM scenarios WHERE network_id = ? LIMIT 1",
        (info["network_id"],),
    ).fetchone()[0]
    source = json.loads(manifest)["source_manifest"]["v4_21_training_dataset"]
    assert source["path"] == str(v4_source_path.resolve())
    assert source["sha256"] == _sha256(v4_source_path.read_bytes())
    assert source["manifest"] == {
        "path": str(v4_manifest_path.resolve()),
        "sha256": _sha256(v4_manifest_path.read_bytes()),
    }


def test_backfill_tributary_features_rejects_missing_network_scoped_run_mapping():
    from swmm_resilience.database.csv_backfill import backfill_tributary_features

    dataset_v21 = _synthetic_dataset_v4_21()
    with pytest.raises(ValueError, match=r"network_id=41.*shape_id='base'"):
        backfill_tributary_features(
            None, dataset_v21, {}, {}, 41
        )
