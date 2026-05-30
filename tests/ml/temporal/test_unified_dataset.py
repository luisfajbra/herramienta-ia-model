"""TDD tests for build_unified_dataset()."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.database.schema import create_schema
from swmm_resilience.ml.temporal.dataset import SWMM_OUTPUT_COLS, build_unified_dataset
from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset

# ── helpers ──────────────────────────────────────────────────────────────────

_CSV_COLS = [
    "run_id", "node_id", "network_hash", "network_file",
    "inflow_multiplier", "scenario_type", "spatial_pattern",
    "invert_elev_m", "full_depth_m", "base_inflow_lps", "node_type",
    "in_degree", "out_degree",
    "upstream_pipes_count", "upstream_diam_max_m", "upstream_diam_min_m",
    "upstream_diam_avg_m", "upstream_slope_avg", "upstream_slope_max",
    "upstream_capacity_lps",
    "downstream_pipes_count", "downstream_diam_max_m", "downstream_diam_min_m",
    "downstream_diam_avg_m", "downstream_slope_avg", "downstream_slope_max",
    "downstream_capacity_lps",
    "max_depth_m", "max_depth_ratio", "time_to_peak_min",
    "depth_rate_m_per_min", "max_total_outflow_lps", "time_to_peak_outflow_min",
    "downstream_link_peak_flows_lps_json",
    "flooded", "peak_flooding_lps", "flooding_duration_min",
]

_PARQUET_COLS = [
    "run_id", "network_hash", "node_id", "step_index",
    "time_sec", "time_min",
    "total_inflow_lps", "lateral_inflow_lps", "depth_m",
    "depth_ratio", "flooding_lps", "total_outflow_lps", "failed_now",
]


def _make_csv(path: Path, n_runs: int = 2, n_nodes: int = 3) -> None:
    rows = []
    network_hash = uuid.uuid4().hex
    for i in range(n_runs):
        run_id = f"run_{i:03d}"
        multiplier = 1.0 + i * 0.25
        for j in range(n_nodes):
            node_id = f"J-{j:03d}"
            flooded = 1 if (j == 0 and i > 0) else 0
            rows.append({
                "run_id": run_id, "node_id": node_id,
                "network_hash": network_hash, "network_file": "test.inp",
                "inflow_multiplier": multiplier,
                "scenario_type": "uniform", "spatial_pattern": "uniform",
                "invert_elev_m": 10.0, "full_depth_m": 1.2,
                "base_inflow_lps": 5.0,
                "node_type": "outfall" if j == 0 else "junction",
                "in_degree": 1, "out_degree": 1,
                "upstream_pipes_count": 1,
                "upstream_diam_max_m": 0.3, "upstream_diam_min_m": 0.3,
                "upstream_diam_avg_m": 0.3,
                "upstream_slope_avg": 0.001, "upstream_slope_max": 0.002,
                "upstream_capacity_lps": 50.0,
                "downstream_pipes_count": 1,
                "downstream_diam_max_m": 0.3, "downstream_diam_min_m": 0.3,
                "downstream_diam_avg_m": 0.3,
                "downstream_slope_avg": 0.001, "downstream_slope_max": 0.002,
                "downstream_capacity_lps": 50.0,
                "max_depth_m": 0.9, "max_depth_ratio": 0.75,
                "time_to_peak_min": 30.0, "depth_rate_m_per_min": 0.01,
                "max_total_outflow_lps": 20.0, "time_to_peak_outflow_min": 35.0,
                "downstream_link_peak_flows_lps_json": "{}",
                "flooded": flooded, "peak_flooding_lps": 8.0 if flooded else 0.0,
                "flooding_duration_min": 15.0 if flooded else 0.0,
            })
    pd.DataFrame(rows, columns=_CSV_COLS).to_csv(path, index=False)


def _make_parquet(directory: Path, run_id: str, network_hash: str,
                  n_nodes: int = 3, n_steps: int = 10) -> Path:
    records = []
    for j in range(n_nodes):
        node_id = f"J-{j:03d}"
        for step in range(n_steps):
            records.append({
                "run_id": run_id, "network_hash": network_hash,
                "node_id": node_id, "step_index": step,
                "time_sec": step * 300, "time_min": float(step * 5),
                "total_inflow_lps": 10.0 + step * 0.5,
                "lateral_inflow_lps": 5.0,
                "depth_m": 0.5, "depth_ratio": 0.3,
                "flooding_lps": 0.0, "total_outflow_lps": 8.0, "failed_now": 0,
            })
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.parquet"
    pd.DataFrame(records, columns=_PARQUET_COLS).to_parquet(path, index=False)
    return path


def _setup(tmp_path: Path, n_runs: int = 2, n_nodes: int = 3):
    csv_path = tmp_path / "dataset_ml.csv"
    _make_csv(csv_path, n_runs=n_runs, n_nodes=n_nodes)

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    network_hash = pd.read_csv(csv_path)["network_hash"].iloc[0]
    parquet_dir = tmp_path / "parquets"

    for i in range(n_runs):
        run_id = f"run_{i:03d}"
        multiplier = 1.0 + i * 0.25
        conn.execute(
            "INSERT INTO runs (run_id, network_file, network_hash, scenario_type, "
            "spatial_pattern, delta_inflow_lps, inflow_multiplier, status, input_source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, "test.inp", network_hash, "uniform", "uniform",
             0.0, multiplier, "done", "test"),
        )
        parquet_path = _make_parquet(parquet_dir, run_id, network_hash, n_nodes=n_nodes)
        conn.execute(
            "INSERT INTO temporal_artifacts (run_id, network_hash, parquet_path) "
            "VALUES (?, ?, ?)",
            (run_id, network_hash, str(parquet_path)),
        )
    conn.commit()
    conn.close()
    return csv_path, db_path


# ── tests ─────────────────────────────────────────────────────────────────────

class TestSampleCount:
    def test_one_sample_per_node_per_run(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert ds.X_seq.shape[0] == 6, f"Expected 6, got {ds.X_seq.shape[0]}"

    def test_no_duplicates(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        pairs = list(zip(ds.meta["run_id"], ds.meta["node_id"]))
        assert len(pairs) == len(set(pairs))


class TestSwmmOutputsDropped:
    def test_swmm_output_cols_not_in_static(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert ds.X_static.shape[1] == 21, (
            f"Expected 21 static features, got {ds.X_static.shape[1]}. "
            "SWMM output columns may have leaked."
        )


class TestOutputShapes:
    def test_x_seq_shape(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        N, T, F = ds.X_seq.shape
        assert N == 6
        assert F == 2, f"Expected 2 temporal features (inflow only), got {F}"
        assert T >= 1

    def test_x_static_shape(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert ds.X_static.shape == (6, 21)

    def test_groups_are_run_ids(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert set(ds.groups) == {"run_000", "run_001"}

    def test_no_nans_in_static(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert not np.isnan(ds.X_static).any(), "NaNs found in X_static"


class TestLabels:
    def test_y_class_binary(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert set(ds.y_class.tolist()).issubset({0, 1})

    def test_flooded_node_labeled_correctly(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        mask = (ds.meta["run_id"] == "run_001") & (ds.meta["node_id"] == "J-000")
        assert ds.y_class[mask].all(), "J-000 in run_001 should be labeled flooded"

    def test_y_reg_nonnegative(self, tmp_path):
        csv_path, db_path = _setup(tmp_path, n_runs=2, n_nodes=3)
        ds = build_unified_dataset(csv_path=csv_path, db_path=db_path)
        assert (ds.y_reg >= 0).all()
