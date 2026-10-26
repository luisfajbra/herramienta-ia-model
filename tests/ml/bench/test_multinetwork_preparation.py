import json
import hashlib
import sqlite3
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

import main
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.csv_backfill import (
    backfill_networks_and_runs,
    backfill_tributary_features,
)
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.ml.bench.schemas import PreparedDataset
from swmm_resilience.ml.bench import preprocess as preprocess_module
from swmm_resilience.ml.bench.train import load_candidate, train_candidate
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21


def _datasets():
    v17_rows = []
    v21_rows = []
    for factor in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        for node_idx in range(4):
            flooded = int(node_idx < 2 and factor >= 2.0)
            identity = {
                "node_id": f"N{node_idx}",
                "shape_id": "base",
                "factor_mult": factor,
                "coord_x": float(node_idx),
                "coord_y": float(node_idx * 2),
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
                "upstream_capacity_lps": 5.0,
                "q_pico_nodo": 2.0 * factor,
                "q_pico_acum_escalado": 10.0 * factor,
                "duracion_horas": 3.0,
                "tiempo_al_pico_h": 0.7,
                "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                "inunda": flooded,
            }
            v17_rows.append(identity)
            v21_rows.append(
                {
                    **identity,
                    "volumen_local_integrado_m3": 1.0 * factor,
                    "volumen_tributario_integrado_m3": 2.0 * factor,
                    "duracion_efectiva_tributario_h": 1.5,
                    "tiempo_primer_pico_tributario_h": 0.5,
                    "tiempo_sobre_capacidad_h": max(factor - 1.0, 0.0),
                    "integral_exceso_capacidad_m3": max(factor - 1.0, 0.0) * 1.2,
                }
            )
    return pd.DataFrame(v17_rows), pd.DataFrame(v21_rows)


def test_bench_prepare_keeps_v17_as_default_contract():
    args = main.build_parser().parse_args(["--bench-prepare"])

    assert args.bench_contract == "tabular_v3_17"


def test_cli_prepares_and_reloads_v4_21_for_two_selected_networks(tmp_path, monkeypatch):
    db_path = tmp_path / "experiment.sqlite3"
    v17, v21 = _datasets()
    network_ids = []
    conn = connect_managed_database(db_path)
    apply_migrations(conn)
    for name in ("A", "B"):
        inp = tmp_path / f"{name}.inp"
        inp.write_text(f"[TITLE]\nnetwork {name}\n", encoding="utf-8")
        v17_source = tmp_path / f"{name}_v17.csv"
        v4_source = tmp_path / f"{name}_v4.csv"
        v4_manifest = tmp_path / f"{name}_v4.csv.manifest.json"
        v17_source.write_text("v17 source evidence", encoding="utf-8")
        v4_source.write_text("v4 source evidence", encoding="utf-8")
        v4_manifest.write_text("validated v4 source manifest", encoding="utf-8")
        info = backfill_networks_and_runs(
            conn,
            v17,
            inp,
            name,
            source_dataset_path=v17_source,
        )
        backfill_tributary_features(
            conn,
            v21,
            info["run_id_by_key"],
            info["node_pk_by_id"],
            info["network_id"],
            source_dataset_path=v4_source,
            source_manifest_path=v4_manifest,
        )
        if name == "B":
            excluded_run_id = info["run_id_by_key"][(info["network_id"], "base", 3.5)]
            conn.execute(
                "UPDATE runs SET continuity_error_pct=6.0 WHERE run_id=?",
                (excluded_run_id,),
            )
        conn.commit()
        network_ids.append(info["network_id"])
    conn.close()

    config = SimpleNamespace(
        dataset=SimpleNamespace(db_path=db_path, flood_threshold_m3=1.0),
        validation=SimpleNamespace(max_continuity_error_pct=5.0),
        bench=SimpleNamespace(
            protocols=("GroupKFold5",),
            families={},
            enabled_families=lambda: [],
        ),
    )
    output_dir = tmp_path / "prepared"
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    monkeypatch.setattr(
        preprocess_module,
        "TemporaryDirectory",
        lambda prefix: nullcontext(snapshot_dir),
    )
    monkeypatch.setattr(main, "load_config", lambda _path: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--bench-prepare",
            "--bench-db",
            str(db_path),
            "--bench-contract",
            "tabular_v4_21",
            "--network-ids",
            ",".join(map(str, network_ids)),
            "--bench-output",
            str(output_dir),
        ],
    )

    main.main()

    prepared_dirs = list(output_dir.iterdir())
    assert len(prepared_dirs) == 1
    prepared = PreparedDataset.load(prepared_dirs[0])
    assert set(prepared.keys["network_id"]) == set(network_ids)
    assert tuple(prepared.X.columns) == FEATURE_COLUMNS_V21
    assert prepared.manifest["contract_id"] == TABULAR_V4_21.contract_id
    assert prepared.manifest["feature_contract_sha256"] == TABULAR_V4_21.descriptor_sha256
    assert prepared.manifest["source_network_ids"] == sorted(network_ids)
    assert len(prepared.manifest["source_networks"]) == 2
    source_runs = [
        run
        for network in prepared.manifest["source_networks"]
        for run in network["runs"]
    ]
    assert all(item["network_sha256"] for item in prepared.manifest["source_networks"])
    assert all(item["inp_path"] for item in prepared.manifest["source_networks"])
    assert all(run["event_series_sha256"] for run in source_runs)
    assert all(run["event_series_components"]["network_inp_sha256"] for run in source_runs)
    assert all(run["v4_21_dataset"]["sha256"] for run in source_runs)
    assert prepared.manifest["excluded_run_ids"] == [excluded_run_id]
    assert prepared.manifest["excluded_runs"][0]["reason"] == "continuity_error_exceeds_threshold"
    assert prepared.manifest["prep_id"] == prepared.prep_id

    # Exercise both intended model families through persisted, verified pairs.
    # Tiny settings keep this a contract/provenance integration test, not tuning.
    model_params = {
        "xgboost": {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.3,
            "subsample": 1.0,
        },
        "mlp": {
            "hidden_sizes": (8,),
            "epochs": 2,
            "learning_rate": 0.01,
            "batch_size": 32,
            "dropout": 0.0,
        },
    }
    for family, params in model_params.items():
        written = train_candidate(
            prepared,
            family,
            params,
            params,
            tmp_path / "candidates",
        )
        loaded = load_candidate(written.classifier_path.parent)
        assert loaded.metadata["prep_id"] == prepared.prep_id
        assert loaded.metadata["feature_contract_id"] == TABULAR_V4_21.contract_id
        assert loaded.metadata["ordered_features"] == list(FEATURE_COLUMNS_V21)
        assert loaded.metadata["source_network_ids"] == prepared.manifest["source_network_ids"]
        assert loaded.metadata["source_networks"] == prepared.manifest["source_networks"]

    # Recompute the event hash after associating a valid-looking event payload
    # with the wrong scenario; the cross-reference must still reject it.
    conn = sqlite3.connect(db_path)
    scenario_id, config_json = conn.execute(
        "SELECT scenario_id, config_json FROM scenarios ORDER BY scenario_id LIMIT 1"
    ).fetchone()
    source_config = json.loads(config_json)
    source = source_config["source_manifest"]
    source["event_series_components"]["shape_id"] = "different-event"
    source["event_series_sha256"] = hashlib.sha256(
        json.dumps(source["event_series_components"], sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()
    updated_config_json = json.dumps(source_config, sort_keys=True)
    updated_config_sha = hashlib.sha256(updated_config_json.encode("utf-8")).hexdigest()
    conn.execute(
        "UPDATE scenarios SET config_json=?, config_sha256=? WHERE scenario_id=?",
        (updated_config_json, updated_config_sha, scenario_id),
    )
    conn.execute(
        "UPDATE runs SET config_sha256=? WHERE scenario_id=?",
        (updated_config_sha, scenario_id),
    )
    conn.commit()
    conn.close()
    conn = sqlite3.connect(db_path)
    selected_run_ids = [row[0] for row in conn.execute("SELECT run_id FROM runs")]
    conn.close()
    with pytest.raises(ValueError, match="Event-series identity does not match"):
        preprocess_module._source_provenance(
            db_path,
            included_run_ids=selected_run_ids,
            run_ids=None,
            network_ids=network_ids,
            max_continuity_error_pct=5.0,
        )

    # A selected multi-network dataset cannot claim reproducibility when one
    # source run has lost its INP/event/dataset manifest.
    conn = sqlite3.connect(db_path)
    scenario_id, config_json = conn.execute(
        "SELECT scenario_id, config_json FROM scenarios ORDER BY scenario_id LIMIT 1"
    ).fetchone()
    source_config = json.loads(config_json)
    source_config.pop("source_manifest", None)
    updated_config_json = json.dumps(source_config, sort_keys=True)
    updated_config_sha = hashlib.sha256(updated_config_json.encode("utf-8")).hexdigest()
    conn.execute(
        "UPDATE scenarios SET config_json=?, config_sha256=? WHERE scenario_id=?",
        (updated_config_json, updated_config_sha, scenario_id),
    )
    conn.execute(
        "UPDATE runs SET config_sha256=? WHERE scenario_id=?",
        (updated_config_sha, scenario_id),
    )
    conn.commit()
    conn.close()
    conn = sqlite3.connect(db_path)
    selected_run_ids = [row[0] for row in conn.execute("SELECT run_id FROM runs")]
    conn.close()
    with pytest.raises(ValueError, match="Source manifest is missing.*multi-network"):
        preprocess_module._source_provenance(
            db_path,
            included_run_ids=selected_run_ids,
            run_ids=None,
            network_ids=network_ids,
            max_continuity_error_pct=5.0,
        )

    manifest_path = prepared_dirs[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_contract_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="prep_id|contract"):
        PreparedDataset.load(prepared_dirs[0])
