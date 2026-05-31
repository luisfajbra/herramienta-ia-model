"""TDD tests for build_temporal_window_summary."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pandas as pd
import pytest

from swmm_resilience.database.schema import create_schema
from swmm_resilience.ml.temporal.predict import (
    TEMPORAL_WINDOW_COLUMNS,
    build_temporal_window_summary,
)


def _populate_regression_tables(db_path: str) -> str:
    """Create regression_windows/regression_runs tables and insert rows."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS regression_windows (
            regression_id TEXT PRIMARY KEY,
            duration_min REAL,
            time_skip_days REAL,
            mean_capacity_lps REAL,
            n_peaks INTEGER,
            n_drains INTEGER,
            n_swales INTEGER,
            n_inlets INTEGER
        );
        CREATE TABLE IF NOT EXISTS regression_runs (
            regression_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            network_hash TEXT NOT NULL,
            PRIMARY KEY (regression_id, run_id)
        );
    """)

    network_hash = uuid.uuid4().hex
    run_id = f"run_reg_{uuid.uuid4().hex[:8]}"

    conn.execute(
        "INSERT INTO runs (run_id, network_file, network_hash, scenario_type, "
        "spatial_pattern, delta_inflow_lps, inflow_multiplier, status, input_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, "test.inp", network_hash, "uniform", "uniform", 0.0, 2.5, "done", "test"),
    )

    reg_id = "reg_001"
    conn.execute(
        "INSERT INTO regression_windows (regression_id, duration_min, time_skip_days, "
        "mean_capacity_lps, n_peaks, n_drains, n_swales, n_inlets) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (reg_id, 120.0, 1.0, 45.0, 3, 2, 1, 0),
    )
    conn.execute(
        "INSERT INTO regression_runs (regression_id, run_id, network_hash) VALUES (?, ?, ?)",
        (reg_id, run_id, network_hash),
    )

    conn.commit()
    conn.close()
    return network_hash


class TestBuildTemporalWindowSummary:
    def test_returns_expected_columns(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        create_schema(conn)
        conn.close()

        network_hash = _populate_regression_tables(db)
        result = build_temporal_window_summary(network_hash=network_hash, db_path=db)
        assert list(result.columns) == TEMPORAL_WINDOW_COLUMNS

    def test_returns_one_row_per_regression(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        create_schema(conn)
        conn.close()

        network_hash = _populate_regression_tables(db)
        result = build_temporal_window_summary(network_hash=network_hash, db_path=db)
        assert len(result) == 1

    def test_returns_empty_when_no_rows(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        create_schema(conn)
        conn.close()

        _populate_regression_tables(db)
        fake_hash = uuid.uuid4().hex
        result = build_temporal_window_summary(network_hash=fake_hash, db_path=db)
        assert result.empty
        assert list(result.columns) == TEMPORAL_WINDOW_COLUMNS

    def test_returns_empty_when_db_missing(self, tmp_path):
        missing = tmp_path / "nonexistent.db"
        result = build_temporal_window_summary(
            network_hash=uuid.uuid4().hex, db_path=missing,
        )
        assert result.empty
        assert list(result.columns) == TEMPORAL_WINDOW_COLUMNS

    def test_returns_empty_when_table_missing(self, tmp_path):
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(db)
        create_schema(conn)
        conn.close()

        result = build_temporal_window_summary(
            network_hash=uuid.uuid4().hex, db_path=db,
        )
        assert result.empty
        assert list(result.columns) == TEMPORAL_WINDOW_COLUMNS
