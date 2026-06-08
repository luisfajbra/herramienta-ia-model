"""Tests for swmm_resilience.ml.scenario_predict.predict_scenario."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml import scenario_predict
from swmm_resilience.validation.hydrograph_csv import HydrographScenario


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_node_scenario():
    """Minimal HydrographScenario with two nodes."""
    return HydrographScenario(
        scenario_id="test_scenario",
        node_series={
            "J1": [(0.0, 0.0), (1.0, 50.0), (2.0, 30.0)],
            "J2": [(0.0, 0.0), (1.0, 80.0), (2.0, 60.0)],
        },
        time_grid_hours=[0.0, 1.0, 2.0],
        last_time_hours=2.0,
    )


def _make_static_df(node_ids: list[str]) -> pd.DataFrame:
    """Build a minimal static feature DataFrame for the given node IDs."""
    from swmm_resilience.ml.trainer import FEATURE_COLS

    rows = []
    for i, nid in enumerate(node_ids):
        row = {"node_id": nid, "coord_x": float(i), "coord_y": float(i)}
        # Populate every FEATURE_COL with a non-zero value so models get
        # numeric input. factor_mult, q_pico_nodo, q_pico_acum_escalado are
        # overridden by compute_dynamic_features.
        for col in FEATURE_COLS:
            if col not in row:
                row[col] = float(i + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_full_df(node_ids: list[str]) -> pd.DataFrame:
    """Static + topology columns expected after compute_topology_features."""
    df = _make_static_df(node_ids)
    # Add topology columns not yet present
    for col in ["dist_outfall_m", "n_nodos_aguas_arriba", "q_pico_acum_base", "upstream_capacity_lps"]:
        if col not in df.columns:
            df[col] = 1.0
    return df


class _FakeClassifier:
    """Returns 0 for J1, 1 for J2 (positional, first two nodes)."""

    def predict(self, X):
        n = len(X)
        return np.array([i % 2 for i in range(n)], dtype=int)


class _FakeRegressor:
    """Returns log1p(10.0) for every row → expm1 gives 10.0."""

    def predict(self, X):
        return np.full(len(X), np.log1p(10.0))


class _FakeNegativeRegressor:
    """Returns negative values to test clipping."""

    def predict(self, X):
        return np.full(len(X), -5.0)


# ---------------------------------------------------------------------------
# Helpers to set up monkeypatches
# ---------------------------------------------------------------------------

def _patch_pipeline(monkeypatch, clf, reg):
    """Monkeypatch joblib.load and feature extraction helpers."""

    def fake_joblib_load(path):
        path = Path(path)
        if "clf" in path.stem or "classifier" in path.stem:
            return clf
        return reg

    monkeypatch.setattr(scenario_predict.joblib, "load", fake_joblib_load)

    def fake_extract_static_features(inp_path):
        return _make_static_df(["J1", "J2"])

    def fake_compute_topology_features(static_df, inp_path):
        return _make_full_df(list(static_df["node_id"]))

    def fake_compute_dynamic_features(full_df, factor):
        node_ids = list(full_df["node_id"])
        return pd.DataFrame(
            {
                "node_id": node_ids,
                "factor_mult": [factor] * len(node_ids),
                "q_pico_nodo": [float(i + 1) * factor for i in range(len(node_ids))],
                "q_pico_acum_escalado": [float(i + 1) * factor for i in range(len(node_ids))],
            }
        )

    monkeypatch.setattr(scenario_predict, "extract_static_features", fake_extract_static_features)
    monkeypatch.setattr(scenario_predict, "compute_topology_features", fake_compute_topology_features)
    monkeypatch.setattr(scenario_predict, "compute_dynamic_features", fake_compute_dynamic_features)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_predict_scenario_returns_correct_columns(monkeypatch, two_node_scenario, tmp_path):
    """Output DataFrame must have node_id, inunda_pred, and vol_pred_m3 columns."""
    clf_path = tmp_path / "clf.joblib"
    reg_path = tmp_path / "reg.joblib"
    clf_path.touch()
    reg_path.touch()

    _patch_pipeline(monkeypatch, _FakeClassifier(), _FakeRegressor())

    result = scenario_predict.predict_scenario(
        scenario=two_node_scenario,
        clf_path=clf_path,
        reg_path=reg_path,
        flood_threshold_m3=1.0,
        inp_path=tmp_path / "network.inp",
    )

    assert set(result.columns) >= {"node_id", "inunda_pred", "vol_pred_m3"}, (
        f"Missing columns. Got: {list(result.columns)}"
    )


def test_predict_scenario_node_ids_match_scenario(monkeypatch, two_node_scenario, tmp_path):
    """Returned node_ids must match the keys in scenario.node_series."""
    clf_path = tmp_path / "clf.joblib"
    reg_path = tmp_path / "reg.joblib"
    clf_path.touch()
    reg_path.touch()

    _patch_pipeline(monkeypatch, _FakeClassifier(), _FakeRegressor())

    result = scenario_predict.predict_scenario(
        scenario=two_node_scenario,
        clf_path=clf_path,
        reg_path=reg_path,
        flood_threshold_m3=1.0,
        inp_path=tmp_path / "network.inp",
    )

    expected_nodes = set(two_node_scenario.node_series.keys())
    returned_nodes = set(result["node_id"].astype(str))
    assert returned_nodes == expected_nodes, (
        f"Expected {expected_nodes}, got {returned_nodes}"
    )


def test_predict_scenario_vol_clipped_to_zero(monkeypatch, two_node_scenario, tmp_path):
    """Negative regressor outputs must be clipped to 0."""
    clf_path = tmp_path / "clf.joblib"
    reg_path = tmp_path / "reg.joblib"
    clf_path.touch()
    reg_path.touch()

    # All nodes predicted flooded so regressor is called for all
    class _AllFloodedClassifier:
        def predict(self, X):
            return np.ones(len(X), dtype=int)

    _patch_pipeline(monkeypatch, _AllFloodedClassifier(), _FakeNegativeRegressor())

    result = scenario_predict.predict_scenario(
        scenario=two_node_scenario,
        clf_path=clf_path,
        reg_path=reg_path,
        flood_threshold_m3=1.0,
        inp_path=tmp_path / "network.inp",
    )

    assert (result["vol_pred_m3"] >= 0).all(), (
        f"vol_pred_m3 must be >= 0 for all rows. Min: {result['vol_pred_m3'].min()}"
    )


def test_predict_scenario_inunda_pred_is_int(monkeypatch, two_node_scenario, tmp_path):
    """inunda_pred column must contain integer values (0 or 1)."""
    clf_path = tmp_path / "clf.joblib"
    reg_path = tmp_path / "reg.joblib"
    clf_path.touch()
    reg_path.touch()

    _patch_pipeline(monkeypatch, _FakeClassifier(), _FakeRegressor())

    result = scenario_predict.predict_scenario(
        scenario=two_node_scenario,
        clf_path=clf_path,
        reg_path=reg_path,
        flood_threshold_m3=1.0,
        inp_path=tmp_path / "network.inp",
    )

    assert pd.api.types.is_integer_dtype(result["inunda_pred"]), (
        f"inunda_pred must be integer dtype, got {result['inunda_pred'].dtype}"
    )
    assert set(result["inunda_pred"].unique()).issubset({0, 1}), (
        f"inunda_pred values must be 0 or 1, got {result['inunda_pred'].unique()}"
    )
