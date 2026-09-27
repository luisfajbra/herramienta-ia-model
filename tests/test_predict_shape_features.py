"""predict_network must label its rows with the .inp's own hydrograph shape.

Regression guard for a bug that survived because it was invisible: with only
19 training shapes, duracion_horas and tiempo_al_pico_h barely mattered, so
feeding the model their 0.0 defaults cost little. At 40 shapes the regressor
ranks them 1st and 3rd by importance, and the same omission underestimated
total flooded volume by up to 94%.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.extraction import dynamic_features as dyn_mod
from swmm_resilience.ml import predict as predict_mod


class _StubModel:
    """Records the feature frame it is asked to predict on."""

    def __init__(self):
        self.seen = []

    def predict(self, X):
        self.seen.append(X.copy())
        return np.zeros(len(X), dtype=int)


@pytest.fixture
def captured_features(monkeypatch, tmp_path):
    """Run predict_network against stubs and hand back the features it built."""
    static = pd.DataFrame(
        {
            "node_id": ["N0", "N1"],
            "base_inflow_lps": [10.0, 5.0],
            "q_pico_acum_base": [10.0, 15.0],
            "coord_x": [0.0, 1.0],
            "coord_y": [0.0, 1.0],
        }
    )

    inp_path = tmp_path / "net.inp"
    inp_path.write_text("dummy", newline="\n")

    clf = _StubModel()
    reg = _StubModel()

    monkeypatch.setattr(predict_mod.joblib, "load", lambda path: clf if "classifier" in str(path) else reg)
    monkeypatch.setattr(predict_mod, "_md5", lambda path: "hash")
    monkeypatch.setattr(predict_mod, "extract_static_features", lambda p: static)
    monkeypatch.setattr(predict_mod, "compute_topology_features", lambda df, p: df)
    monkeypatch.setattr(predict_mod, "base_shape_stats", lambda p: (4.25, 1.5))
    monkeypatch.setattr(predict_mod, "FEATURE_COLS", ["duracion_horas", "tiempo_al_pico_h"])

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "training_inp_hash.txt").write_text("hash", newline="\n")

    config = type(
        "Cfg",
        (),
        {
            "network": type("N", (), {"inp_path": inp_path})(),
            "simulation": type("S", (), {"factor_min": 0.2, "factor_max": 5.0})(),
        },
    )()

    predict_mod.predict_network(2.0, config, models_dir)
    return clf.seen[0]


def test_shape_descriptors_reach_the_model(captured_features):
    assert captured_features["duracion_horas"].unique().tolist() == [4.25]
    assert captured_features["tiempo_al_pico_h"].unique().tolist() == [1.5]


def test_shape_descriptors_are_never_the_zero_default(captured_features):
    """(0.0, 0.0) matches no training row; it must never reach the model."""
    zeros = (captured_features["duracion_horas"] == 0.0) & (
        captured_features["tiempo_al_pico_h"] == 0.0
    )

    assert not zeros.any()


def test_compute_dynamic_features_still_defaults_to_zero():
    """The default itself is fine — callers omitting it were the problem."""
    frame = dyn_mod.compute_dynamic_features(
        pd.DataFrame(
            {"node_id": ["N0"], "base_inflow_lps": [1.0], "q_pico_acum_base": [1.0]}
        ),
        1.0,
    )

    assert frame["duracion_horas"].item() == 0.0
    assert frame["tiempo_al_pico_h"].item() == 0.0


def test_base_shape_stats_without_timeseries_returns_zeros(monkeypatch):
    monkeypatch.setattr(predict_mod, "load_inp", lambda p: {})

    assert predict_mod.base_shape_stats(Path("x.inp")) == (0.0, 0.0)


def test_base_shape_stats_reads_the_inp_timeseries(monkeypatch):
    class _TS:
        data = [(0.0, 0.0), (1.5, 9.0), (3.0, 0.0)]

    monkeypatch.setattr(predict_mod, "load_inp", lambda p: {"TIMESERIES": {"t": _TS()}})

    assert predict_mod.base_shape_stats(Path("x.inp")) == (3.0, 1.5)
