"""Tests for swmm_resilience.ml.scenario_predict (ScenarioPredictor)."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.extraction.dynamic_features import (
    compute_scenario_dynamic_features,
)
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


_NETWORK_NODES = ["J1", "J2", "J3"]  # J3 has no direct inflow in the scenario


def _make_full_df(node_ids: list[str]) -> pd.DataFrame:
    """Static + topology columns expected after compute_topology_features."""
    from swmm_resilience.ml.trainer import FEATURE_COLS

    rows = []
    for i, nid in enumerate(node_ids):
        row = {"node_id": nid, "coord_x": float(i), "coord_y": float(i)}
        for col in FEATURE_COLS:
            if col not in row:
                row[col] = float(i + 1)
        # Explicit base inflows: J1=10, J2=10, J3=0 (no direct inflow)
        row["base_inflow_lps"] = 10.0 if nid in ("J1", "J2") else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _make_graph() -> nx.DiGraph:
    """J1 → J2 → J3 chain."""
    g = nx.DiGraph()
    g.add_edges_from([("J1", "J2"), ("J2", "J3")])
    return g


class _FakeClassifier:
    """Alternates 0/1 by row position; no predict_proba (tests fallback)."""

    def predict(self, X):
        return np.array([i % 2 for i in range(len(X))], dtype=int)


class _FakeRegressor:
    """Returns log1p(10.0) for every row → expm1 gives 10.0."""

    def predict(self, X):
        return np.full(len(X), np.log1p(10.0))


class _FakeNegativeRegressor:
    def predict(self, X):
        return np.full(len(X), -5.0)


def _patch_pipeline(monkeypatch, clf, reg, load_counter: list | None = None):
    """Monkeypatch joblib.load and feature extraction helpers."""

    def fake_joblib_load(path):
        if load_counter is not None:
            load_counter.append(str(path))
        path = Path(path)
        if "clf" in path.stem or "classifier" in path.stem:
            return clf
        return reg

    monkeypatch.setattr(scenario_predict.joblib, "load", fake_joblib_load)
    monkeypatch.setattr(
        scenario_predict, "extract_static_features", lambda inp_path: _make_full_df(_NETWORK_NODES)
    )
    monkeypatch.setattr(
        scenario_predict,
        "compute_topology_features",
        lambda static_df, inp_path: static_df,
    )
    monkeypatch.setattr(scenario_predict, "load_inp", lambda path: object())
    monkeypatch.setattr(
        scenario_predict, "build_network_graph", lambda inp: (_make_graph(), set())
    )


def _make_predictor(monkeypatch, tmp_path, clf=None, reg=None, factor_range=None, load_counter=None):
    clf_path = tmp_path / "clf.joblib"
    reg_path = tmp_path / "reg.joblib"
    clf_path.touch()
    reg_path.touch()
    _patch_pipeline(
        monkeypatch, clf or _FakeClassifier(), reg or _FakeRegressor(), load_counter
    )
    return scenario_predict.ScenarioPredictor(
        clf_path=clf_path,
        reg_path=reg_path,
        inp_path=tmp_path / "network.inp",
        factor_range=factor_range,
    )


# ---------------------------------------------------------------------------
# compute_scenario_dynamic_features
# ---------------------------------------------------------------------------

def test_scenario_dynamic_features_accumulate_real_peaks():
    full_df = _make_full_df(_NETWORK_NODES)
    peaks = {"J1": 50.0, "J2": 80.0}
    df = compute_scenario_dynamic_features(full_df, peaks, _make_graph())
    by_node = df.set_index("node_id")

    assert by_node.loc["J1", "q_pico_nodo"] == 50.0
    assert by_node.loc["J3", "q_pico_nodo"] == 0.0  # no direct inflow
    assert by_node.loc["J1", "q_pico_acum_escalado"] == 50.0
    assert by_node.loc["J2", "q_pico_acum_escalado"] == 130.0  # J1 + J2
    assert by_node.loc["J3", "q_pico_acum_escalado"] == 130.0  # upstream only


# ---------------------------------------------------------------------------
# ScenarioPredictor
# ---------------------------------------------------------------------------

def test_predict_returns_contract_columns(monkeypatch, two_node_scenario, tmp_path):
    predictor = _make_predictor(monkeypatch, tmp_path)
    result = predictor.predict(two_node_scenario)
    assert set(result.columns) >= {
        "node_id", "inunda_pred", "prob_inunda", "vol_pred_m3", "extrapolated"
    }, f"Missing columns. Got: {list(result.columns)}"


def test_predict_covers_all_network_junctions(monkeypatch, two_node_scenario, tmp_path):
    """Junctions without direct inflow (J3) must also be predicted."""
    predictor = _make_predictor(monkeypatch, tmp_path)
    result = predictor.predict(two_node_scenario)
    assert set(result["node_id"]) == set(_NETWORK_NODES)


def test_predict_vol_clipped_to_zero(monkeypatch, two_node_scenario, tmp_path):
    class _AllFloodedClassifier:
        def predict(self, X):
            return np.ones(len(X), dtype=int)

    predictor = _make_predictor(
        monkeypatch, tmp_path, clf=_AllFloodedClassifier(), reg=_FakeNegativeRegressor()
    )
    result = predictor.predict(two_node_scenario)
    assert (result["vol_pred_m3"] >= 0).all()


def test_predict_inunda_pred_is_int(monkeypatch, two_node_scenario, tmp_path):
    predictor = _make_predictor(monkeypatch, tmp_path)
    result = predictor.predict(two_node_scenario)
    assert pd.api.types.is_integer_dtype(result["inunda_pred"])
    assert set(result["inunda_pred"].unique()).issubset({0, 1})


def test_extrapolation_flags(monkeypatch, two_node_scenario, tmp_path):
    """J1: peak 50 / base 10 = 5.0 → inside [0.2, 5.0]. J2: 80/10 = 8 → out.
    J3: base 0, no direct peak → never flagged."""
    predictor = _make_predictor(monkeypatch, tmp_path, factor_range=(0.2, 5.0))
    result = predictor.predict(two_node_scenario).set_index("node_id")
    assert not result.loc["J1", "extrapolated"]
    assert result.loc["J2", "extrapolated"]
    assert not result.loc["J3", "extrapolated"]


def test_models_loaded_once_across_predictions(monkeypatch, two_node_scenario, tmp_path):
    loads: list[str] = []
    predictor = _make_predictor(monkeypatch, tmp_path, load_counter=loads)
    predictor.predict(two_node_scenario)
    predictor.predict(two_node_scenario)
    assert len(loads) == 2  # classifier + regressor, constructor only


def test_predict_timed_returns_timings(monkeypatch, two_node_scenario, tmp_path):
    predictor = _make_predictor(monkeypatch, tmp_path)
    _, timings = predictor.predict_timed(two_node_scenario)
    assert set(timings) == {"t_features_s", "t_inference_s"}
    assert all(v >= 0 for v in timings.values())
    assert predictor.model_load_s >= 0
    assert predictor.static_features_s >= 0


def test_predict_scenario_wrapper_compatible(monkeypatch, two_node_scenario, tmp_path):
    """The one-shot functional API keeps working."""
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
    assert set(result.columns) >= {"node_id", "inunda_pred", "vol_pred_m3"}
