"""Tests for the inflow-closure wiring inside swmm_resilience.simulation.batch."""

from __future__ import annotations

import datetime
import json
from types import SimpleNamespace

import pytest

from swmm_resilience.simulation import batch as batch_mod
from swmm_resilience.simulation import timeseries_scenario as ts_mod


class _FakeTs:
    def __init__(self, data):
        self.data = list(data)


class _FakeInflow:
    def __init__(self, ts_name):
        self.time_series = ts_name


class _FakeInp:
    def __init__(self):
        self._s = {
            "INFLOWS": {("1C", "FLOW"): _FakeInflow("1C")},
            "TIMESERIES": {"1C": _FakeTs([(0.0, 0.0)])},
            "OPTIONS": {
                "START_DATE": datetime.date(2025, 1, 4),
                "START_TIME": datetime.time(0, 0, 0),
                "END_DATE": datetime.date(2025, 1, 4),
                "END_TIME": "3:00:00",
            },
        }
        self.written_to = None

    def __contains__(self, key):
        return key in self._s

    def __getitem__(self, key):
        return self._s[key]

    def write_file(self, path):
        self.written_to = str(path)


def _make_config(tmp_path):
    return SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
    )


def test_prepare_closed_scenario_inp_closes_a_shape_that_does_not_end_at_zero(
    monkeypatch, tmp_path,
):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.3)]  # q_norm no termina en 0
    time_grid = [t for t, _ in shape]
    config = _make_config(tmp_path)

    scenario_inp, rpt_path = batch_mod._prepare_closed_scenario_inp(
        config, "storm_x", 1.0, shape, time_grid, {"1C": 2.0}, tmp_path / "run",
    )

    data = fake_inp["TIMESERIES"]["1C"].data
    assert data[2] == pytest.approx((0.1, 0.6))     # ultimo punto original: 0.3 * 2.0 * 1.0
    assert data[3] == pytest.approx((0.15, 0.0))    # punto de cierre, paso de malla 0.05
    assert data[4] == pytest.approx((6.15, 0.0))    # punto de drenaje, plano en cero
    assert len(data) == 5
    assert rpt_path == scenario_inp.with_suffix(".rpt")

    manifest_path = tmp_path / "run" / "storm_x" / "storm_x_f1.000.closure_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["changed_nodes"] == ["1C"]


def test_prepare_closed_scenario_inp_warns_when_closure_changes_a_node(
    monkeypatch, tmp_path,
):
    """Finding 2: spec section 6 requires a warning of the change; this is
    the other production call site of apply_closure_policy besides
    hydrograph_batch.py's close branch."""
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.3)]  # q_norm no termina en 0
    time_grid = [t for t, _ in shape]
    config = _make_config(tmp_path)

    with pytest.warns(UserWarning, match="storm_x"):
        batch_mod._prepare_closed_scenario_inp(
            config, "storm_x", 1.0, shape, time_grid, {"1C": 2.0}, tmp_path / "run",
        )


def test_prepare_closed_scenario_inp_manifest_has_mode_close(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.3)]  # q_norm no termina en 0
    time_grid = [t for t, _ in shape]
    config = _make_config(tmp_path)

    batch_mod._prepare_closed_scenario_inp(
        config, "storm_z", 1.0, shape, time_grid, {"1C": 2.0}, tmp_path / "run",
    )

    manifest_path = tmp_path / "run" / "storm_z" / "storm_z_f1.000.closure_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "close"


def test_prepare_closed_scenario_inp_leaves_an_already_closed_shape_untouched(
    monkeypatch, tmp_path,
):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]  # ya termina en 0
    time_grid = [t for t, _ in shape]
    config = _make_config(tmp_path)

    import warnings as warnings_mod

    with warnings_mod.catch_warnings():
        warnings_mod.simplefilter("error")
        batch_mod._prepare_closed_scenario_inp(
            config, "storm_y", 1.0, shape, time_grid, {"1C": 2.0}, tmp_path / "run",
        )

    data = fake_inp["TIMESERIES"]["1C"].data
    assert data[2] == pytest.approx((0.1, 0.0))   # sin punto de cierre añadido
    assert data[3] == pytest.approx((6.1, 0.0))   # solo el punto de drenaje
    assert len(data) == 4

    manifest_path = tmp_path / "run" / "storm_y" / "storm_y_f1.000.closure_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["changed_nodes"] == []
    assert payload["mode"] == "close"
