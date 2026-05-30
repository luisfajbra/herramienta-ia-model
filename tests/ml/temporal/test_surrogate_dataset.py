# tests/ml/temporal/test_surrogate_dataset.py
"""TDD tests for build_surrogate_dataset()."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.database.schema import create_schema
from swmm_resilience.ml.temporal.dataset import build_surrogate_dataset
from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset

# ── helpers ──────────────────────────────────────────────────────────────────

_PARQUET_COLS = [
    "run_id", "network_hash", "node_id", "step_index",
    "time_sec", "time_min",
    "total_inflow_lps", "lateral_inflow_lps", "depth_m",
    "depth_ratio", "flooding_lps", "total_outflow_lps", "failed_now",
]


def _make_parquet(
    directory: Path,
    run_id: str,
    network_hash: str,
    n_nodes: int = 3,
    n_steps: int = 10,
    flooding_node_idx: int | None = 0,
) -> Path:
    records = []
    for node_idx in range(n_nodes):
        node_id = f"J-{node_idx:03d}"
        flooded = node_idx == flooding_node_idx
        for step in range(n_steps):
            records.append({
                "run_id": run_id,
                "network_hash": network_hash,
                "node_id": node_id,
                "step_index": step,
                "time_sec": step * 300,
                "time_min": float(step * 5),
                "total_inflow_lps": 10.0 + step * 0.5,
                "lateral_inflow_lps": 5.0,
                "depth_m": 0.5 + step * 0.01,
                "depth_ratio": 0.3,
                "flooding_lps": 8.0 if flooded and step >= 5 else 0.0,
                "total_outflow_lps": 8.0,
                "failed_now": 1 if flooded and step >= 5 else 0,
            })
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.parquet"
    pd.DataFrame(records, columns=_PARQUET_COLS).to_parquet(path, index=False)
    return path


def _setup_db(tmp_path: Path, n_runs: int = 2, n_nodes: int = 3) -> tuple[Path, str]:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    network_hash = uuid.uuid4().hex

    parquet_dir = tmp_path / "parquets"

    for i in range(n_runs):
        run_id = f"run_{i:03d}"
        multiplier = 1.0 + i * 0.25
        conn.execute(
            "INSERT INTO runs (run_id, network_file, network_hash, scenario_type, "
            "spatial_pattern, delta_inflow_lps, inflow_multiplier, status, input_source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, "test.inp", network_hash, "uniform", "uniform", 0.0, multiplier, "done", "test"),
        )
        parquet_path = _make_parquet(parquet_dir, run_id, network_hash, n_nodes=n_nodes)
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
    return db_path, network_hash


# ── tests ─────────────────────────────────────────────────────────────────────

class TestOneSamplePerNodePerRun:
    def test_sample_count(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        # 2 runs × 3 nodes = 6 samples
        assert ds.X_seq.shape[0] == 6, f"Expected 6 samples, got {ds.X_seq.shape[0]}"

    def test_no_duplicate_samples(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        pairs = list(zip(ds.meta["run_id"], ds.meta["node_id"]))
        assert len(pairs) == len(set(pairs)), "Duplicate (run_id, node_id) pairs found"


class TestOutputShapes:
    def test_x_seq_shape(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        N, T, F = ds.X_seq.shape
        assert N == 6
        assert F == 6, f"Expected 6 temporal features, got {F}"
        assert T >= 1, "Sequence length must be at least 1"

    def test_x_static_shape(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert ds.X_static.shape == (6, 7), f"Expected (6, 7), got {ds.X_static.shape}"

    def test_labels_shape(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert ds.y_class.shape == (6,)
        assert ds.y_reg.shape == (6,)

    def test_groups_are_run_ids(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert set(ds.groups) == {"run_000", "run_001"}


class TestLabels:
    def test_y_class_is_binary(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert set(ds.y_class.tolist()).issubset({0, 1})

    def test_flooded_node_has_y_class_1(self, tmp_path):
        """Node J-000 floods in both runs (flooding_node_idx=0 by default)."""
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        flooded_mask = ds.meta["node_id"] == "J-000"
        assert ds.y_class[flooded_mask].all(), "J-000 should be labeled flooded"

    def test_nonflooded_node_has_y_class_0(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        dry_mask = ds.meta["node_id"] == "J-002"
        assert not ds.y_class[dry_mask].any(), "J-002 should not be labeled flooded"

    def test_y_reg_nonnegative(self, tmp_path):
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path)
        assert (ds.y_reg >= 0).all()


class TestNoTemporalMode:
    def test_multiplier_appended_to_static(self, tmp_path):
        """use_temporal=False: X_static has 8 cols (7 static + 1 multiplier)."""
        db_path, _ = _setup_db(tmp_path, n_runs=2, n_nodes=3)
        ds = build_surrogate_dataset(db_path=db_path, use_temporal=False)
        assert ds.X_static.shape == (6, 8), f"Expected (6, 8), got {ds.X_static.shape}"
