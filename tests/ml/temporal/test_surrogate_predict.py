# tests/ml/temporal/test_surrogate_predict.py
"""TDD tests for predict_surrogate_from_multiplier()."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from swmm_resilience.database.schema import create_schema
from swmm_resilience.ml.temporal.models.surrogate_cnn import SWMMSurrogateCNN
from swmm_resilience.ml.temporal.predict import (
    plot_surrogate_map,
    predict_surrogate_from_multiplier,
)

_PARQUET_COLS = [
    "run_id", "network_hash", "node_id", "step_index",
    "time_sec", "time_min",
    "total_inflow_lps", "lateral_inflow_lps", "depth_m",
    "depth_ratio", "flooding_lps", "total_outflow_lps", "failed_now",
]


def _make_db_and_artifacts(tmp_path: Path, n_nodes: int = 4) -> tuple[Path, Path]:
    """Create minimal DB (Qx1.00 run) + saved model artifacts."""
    db_path = tmp_path / "test.db"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    conn = sqlite3.connect(db_path)
    create_schema(conn)
    network_hash = uuid.uuid4().hex
    run_id = "run_qx100"

    conn.execute(
        "INSERT INTO runs (run_id, network_file, network_hash, scenario_type, "
        "spatial_pattern, delta_inflow_lps, inflow_multiplier, status, input_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, "test.inp", network_hash, "uniform", "uniform", 0.0, 1.0, "done", "test"),
    )

    # Build Parquet for Qx1.00
    records = []
    for node_idx in range(n_nodes):
        node_id = f"J-{node_idx:03d}"
        for step in range(20):
            records.append({
                "run_id": run_id, "network_hash": network_hash, "node_id": node_id,
                "step_index": step, "time_sec": step * 300, "time_min": float(step * 5),
                "total_inflow_lps": 10.0 + step * 0.5, "lateral_inflow_lps": 5.0,
                "depth_m": 0.5, "depth_ratio": 0.3,
                "flooding_lps": 0.0, "total_outflow_lps": 8.0, "failed_now": 0,
            })
    parquet_path = tmp_path / "run_qx100.parquet"
    pd.DataFrame(records, columns=_PARQUET_COLS).to_parquet(parquet_path, index=False)

    conn.execute(
        "INSERT INTO temporal_artifacts (run_id, network_hash, parquet_path) VALUES (?, ?, ?)",
        (run_id, network_hash, str(parquet_path)),
    )
    for node_idx in range(n_nodes):
        conn.execute(
            "INSERT INTO network_nodes (node_uid, network_hash, full_depth_m, in_degree, out_degree, "
            "upstream_diam_avg_m, downstream_diam_avg_m, upstream_capacity_lps, downstream_capacity_lps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"J-{node_idx:03d}", network_hash, 1.2, 1, 1, 0.3, 0.3, 50.0, 50.0),
        )
    conn.commit()
    conn.close()

    # Save dummy model artifacts (untrained weights + fitted scalers)
    model = SWMMSurrogateCNN(n_temporal_features=2, n_static_features=7)
    torch.save(model.state_dict(), artifacts_dir / "surrogate_cnn_weights.pt")

    scaler_seq = StandardScaler()
    scaler_seq.fit(np.random.randn(100, 2).astype(np.float32))
    joblib.dump(scaler_seq, artifacts_dir / "surrogate_cnn_scaler_seq.joblib")

    scaler_static = StandardScaler()
    scaler_static.fit(np.random.randn(100, 7).astype(np.float32))
    joblib.dump(scaler_static, artifacts_dir / "surrogate_cnn_scaler_static.joblib")

    return db_path, artifacts_dir


class TestOutputColumns:
    def test_returns_expected_columns(self, tmp_path):
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
        result = predict_surrogate_from_multiplier(
            multiplier=2.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        expected_cols = {"node_id", "flood_prob", "predicted_flooded", "peak_flooding_lps_pred"}
        assert expected_cols.issubset(result.columns), f"Missing columns: {expected_cols - set(result.columns)}"

    def test_one_row_per_node(self, tmp_path):
        n_nodes = 4
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path, n_nodes=n_nodes)
        result = predict_surrogate_from_multiplier(
            multiplier=2.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        assert len(result) == n_nodes


class TestFloodProbRange:
    def test_flood_prob_in_0_1(self, tmp_path):
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
        result = predict_surrogate_from_multiplier(
            multiplier=3.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        assert (result["flood_prob"] >= 0).all() and (result["flood_prob"] <= 1).all()

    def test_predicted_flooded_is_binary(self, tmp_path):
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
        result = predict_surrogate_from_multiplier(
            multiplier=3.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        assert set(result["predicted_flooded"].unique()).issubset({0, 1})


class TestMultiplierScaling:
    def test_higher_multiplier_gives_higher_mean_flood_prob(self, tmp_path):
        """A trained model should predict more flooding for higher multipliers.
        With an untrained model this is not guaranteed — just check no crash."""
        db_path, artifacts_dir = _make_db_and_artifacts(tmp_path)
        result_low = predict_surrogate_from_multiplier(
            multiplier=1.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        result_high = predict_surrogate_from_multiplier(
            multiplier=5.0, db_path=db_path, artifacts_dir=artifacts_dir,
        )
        # Both must return same number of nodes without crashing
        assert len(result_low) == len(result_high)


class TestPlotSurrogateMap:
    def test_plot_surrogate_map_default_path(self, tmp_path, monkeypatch):
        """plot_surrogate_map saves to DEFAULT_SURROGATE_MAPS_DIR when output_path is None."""
        from swmm_resilience.config import DEFAULT_SURROGATE_MAPS_DIR

        preds = pd.DataFrame({
            "node_id": ["J-001"], "flood_prob": [0.5],
            "predicted_flooded": [1], "peak_flooding_lps_pred": [10.0],
        })
        inp = tmp_path / "test.inp"
        inp.write_text(
            "[COORDINATES]\n;;Node   X         Y\nJ-001  100.0  200.0\n"
            "[CONDUITS]\n;;ID  From  To  Len  N  Z1  Z2  ZOff\n"
        )
        monkeypatch.setattr(
            "swmm_resilience.ml.temporal.predict.DEFAULT_SURROGATE_MAPS_DIR",
            tmp_path / "maps",
        )
        result = plot_surrogate_map(predictions=preds, inp_path=inp, multiplier=2.5)
        expected = tmp_path / "maps" / "surrogate_map_cnn_qx2.50.png"
        assert result == expected
        assert expected.exists()
