from pathlib import Path

import pandas as pd
import pytest

from swmm_resilience.config import load_config
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.csv_backfill import (
    backfill_networks_and_runs,
    persist_training_run,
)
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


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

    with pytest.raises(ValueError, match="No hay filas inundadas"):
        persist_training_run(
            migrated_managed_conn, df, info["run_id_by_key"], info["node_pk_by_id"], config
        )

    assert migrated_managed_conn.execute(
        "SELECT COUNT(*) FROM training_runs"
    ).fetchone()[0] == 0
    assert migrated_managed_conn.execute(
        "SELECT COUNT(*) FROM model_evaluations"
    ).fetchone()[0] == 0


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
