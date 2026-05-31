"""Tests for build_temporal_windows() — written before implementation (TDD)."""

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.database.schema import create_schema
from swmm_resilience.ml.temporal.dataset import (
    PRE_SWMM_TEMPORAL_COLS,
    STATIC_COLS,
    SWMM_OUTPUT_TEMPORAL_COLS,
    TEMPORAL_COLS,
    build_temporal_windows,
)
from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset, TemporalWindowSpec

# ── shared fixture helpers ────────────────────────────────────────────────────

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
    n_nodes: int,
    n_steps: int,
    flooding_step: int | None = None,
) -> Path:
    """Synthetic Parquet: n_nodes × n_steps rows at 5-min intervals."""
    records = []
    for node_idx in range(n_nodes):
        node_id = f"J-{node_idx:03d}"
        for step in range(n_steps):
            flooding = 5.0 if flooding_step is not None and step == flooding_step else 0.0
            records.append({
                "run_id": run_id,
                "network_hash": network_hash,
                "node_id": node_id,
                "step_index": step,
                "time_sec": step * 300,
                "time_min": float(step * 5),
                "total_inflow_lps": 10.0 + step * 0.1,
                "lateral_inflow_lps": 5.0,
                "depth_m": 0.5 + step * 0.01,
                "depth_ratio": 0.3,
                "flooding_lps": flooding,
                "total_outflow_lps": 8.0,
                "failed_now": 1 if flooding > 0 else 0,
            })
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.parquet"
    pd.DataFrame(records, columns=_PARQUET_COLS).to_parquet(path, index=False)
    return path


def _insert_run(conn: sqlite3.Connection, run_id: str, network_hash: str) -> None:
    conn.execute(
        """INSERT INTO runs
           (run_id, network_file, network_hash, scenario_type, spatial_pattern,
            delta_inflow_lps, inflow_multiplier, input_source, executed_at, status)
           VALUES (?, 'test.inp', ?, 'hydrograph', 'uniform', 0.0, 1.0,
                   'hydrograph', datetime('now'), 'completed')""",
        (run_id, network_hash),
    )


def _insert_artifact(
    conn: sqlite3.Connection,
    run_id: str,
    network_hash: str,
    parquet_path: Path,
    n_nodes: int,
    n_steps: int,
) -> None:
    conn.execute(
        """INSERT INTO temporal_artifacts
           (artifact_id, run_id, network_hash, parquet_path,
            node_count, step_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
        (str(uuid.uuid4()), run_id, network_hash, str(parquet_path), n_nodes, n_steps),
    )


def _insert_nodes(conn: sqlite3.Connection, network_hash: str, n_nodes: int) -> None:
    for i in range(n_nodes):
        conn.execute(
            """INSERT INTO network_nodes
               (network_hash, node_uid, full_depth_m, in_degree, out_degree,
                upstream_diam_avg_m, downstream_diam_avg_m,
                upstream_capacity_lps, downstream_capacity_lps)
               VALUES (?, ?, 2.0, 2, 1, 0.3, 0.25, 50.0, 40.0)""",
            (network_hash, f"J-{i:03d}"),
        )


def _make_db(
    tmp_path: Path,
    *,
    run_id: str,
    network_hash: str,
    parquet_path: Path,
    n_nodes: int,
    n_steps: int,
) -> Path:
    """Single-run SQLite DB with temporal_artifact + network_nodes rows."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    _insert_run(conn, run_id, network_hash)
    _insert_artifact(conn, run_id, network_hash, parquet_path, n_nodes, n_steps)
    _insert_nodes(conn, network_hash, n_nodes)
    conn.commit()
    conn.close()
    return db_path


_SPEC = TemporalWindowSpec(window_min=20, horizon_min=5, step_min=5, resample_min=5)

# ── tests ─────────────────────────────────────────────────────────────────────


class TestWindowShape:
    def test_build_windows_produces_correct_shape(self, tmp_path):
        """2 nodes × 20 steps → X_seq is [N, 4, 6], X_static is [N, 7]."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_shape"
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=2, n_steps=20)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=2, n_steps=20,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        assert ds.X_seq.ndim == 3
        assert ds.X_seq.shape[1] == 4   # window_min / resample_min = 20 / 5
        assert ds.X_seq.shape[2] == 6   # 6 temporal features
        assert ds.X_static.ndim == 2
        assert ds.X_static.shape[1] == 7  # 7 static features
        n = ds.X_seq.shape[0]
        assert ds.y_class.shape == (n,)
        assert ds.y_reg.shape == (n,)
        assert ds.groups.shape == (n,)
        assert len(ds.meta) == n


class TestNoLeakage:
    def test_no_leakage_between_runs(self, tmp_path):
        """Two run_ids → groups contains exactly 2 distinct values."""
        network_hash = "hash_leakage"
        run_id_1 = str(uuid.uuid4())
        run_id_2 = str(uuid.uuid4())
        pq1 = _make_parquet(tmp_path / "r1", run_id_1, network_hash, n_nodes=1, n_steps=20)
        pq2 = _make_parquet(tmp_path / "r2", run_id_2, network_hash, n_nodes=1, n_steps=20)

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        create_schema(conn)
        for run_id, pq in [(run_id_1, pq1), (run_id_2, pq2)]:
            _insert_run(conn, run_id, network_hash)
            _insert_artifact(conn, run_id, network_hash, pq, 1, 20)
        _insert_nodes(conn, network_hash, 1)
        conn.commit()
        conn.close()

        ds = build_temporal_windows(db_path=db_path, window_spec=_SPEC)

        assert ds.X_seq.shape[0] > 0
        assert len(set(ds.groups)) == 2


class TestIncompleteWindow:
    def test_incomplete_window_discarded(self, tmp_path):
        """3 steps × 5 min = 15 min < window_min=20 → 0 samples, no error."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_incomplete"
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=1, n_steps=3)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=1, n_steps=3,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        assert ds.X_seq.shape[0] == 0


class TestHorizonLabel:
    def test_failure_within_horizon_label(self, tmp_path):
        """Flooding at step 4 → y_class[0] == 1 (first window sees it in horizon)."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_horizon_pos"
        # window = steps 0-3, horizon = step 4 (where flooding_step=4)
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=1, n_steps=20, flooding_step=4)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=1, n_steps=20,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        assert int(ds.y_class[0]) == 1

    def test_no_failure_within_horizon_label(self, tmp_path):
        """No flooding anywhere → all y_class == 0."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_horizon_neg"
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=1, n_steps=20)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=1, n_steps=20,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        assert all(int(y) == 0 for y in ds.y_class)


class TestStaticFeatures:
    def test_static_features_joined_correctly(self, tmp_path):
        """X_static values match what was inserted into network_nodes."""
        run_id = str(uuid.uuid4())
        network_hash = "hash_static"
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=1, n_steps=20)
        db = _make_db(
            tmp_path,
            run_id=run_id, network_hash=network_hash,
            parquet_path=pq, n_nodes=1, n_steps=20,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        # _insert_nodes sets: full_depth_m=2.0, in_degree=2, out_degree=1,
        # upstream_diam_avg_m=0.3, downstream_diam_avg_m=0.25,
        # upstream_capacity_lps=50.0, downstream_capacity_lps=40.0
        expected = np.array([2.0, 2.0, 1.0, 0.3, 0.25, 50.0, 40.0], dtype=np.float32)
        np.testing.assert_allclose(ds.X_static[0], expected)


class TestFeatureContracts:
    def test_deployable_temporal_features_exclude_swmm_outputs(self):
        assert PRE_SWMM_TEMPORAL_COLS == ["total_inflow_lps", "lateral_inflow_lps"]
        forbidden = set(SWMM_OUTPUT_TEMPORAL_COLS)
        assert forbidden
        assert not (set(PRE_SWMM_TEMPORAL_COLS) & forbidden)

    def test_legacy_temporal_cols_are_explicitly_post_swmm(self):
        assert "flooding_lps" in SWMM_OUTPUT_TEMPORAL_COLS
        assert "depth_m" in SWMM_OUTPUT_TEMPORAL_COLS
        assert "total_outflow_lps" in SWMM_OUTPUT_TEMPORAL_COLS
        assert TEMPORAL_COLS == PRE_SWMM_TEMPORAL_COLS + SWMM_OUTPUT_TEMPORAL_COLS

    def test_window_dataset_meta_records_feature_names(self, tmp_path):
        run_id = str(uuid.uuid4())
        network_hash = "hash_contract"
        pq = _make_parquet(tmp_path, run_id, network_hash, n_nodes=1, n_steps=20)
        db = _make_db(
            tmp_path,
            run_id=run_id,
            network_hash=network_hash,
            parquet_path=pq,
            n_nodes=1,
            n_steps=20,
        )

        ds = build_temporal_windows(db_path=db, window_spec=_SPEC)

        assert ds.meta.attrs["temporal_feature_names"] == TEMPORAL_COLS
        assert ds.meta.attrs["static_feature_names"] == STATIC_COLS
