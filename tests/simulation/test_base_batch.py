from __future__ import annotations

import datetime
import sys
from types import SimpleNamespace

import pytest

from swmm_resilience.simulation import batch


class _Inp:
    def __init__(self):
        self.data = {
            "OPTIONS": {
                "FLOW_UNITS": "LPS",
                "START_DATE": datetime.date(2025, 1, 1),
                "START_TIME": datetime.time(0, 0),
            },
            "INFLOWS": {
                ("J1", "FLOW"): SimpleNamespace(
                    time_series="TS1", scale_factor=2.0,
                    base_value=5.0, pattern=None, mass_unit_factor=1.0,
                ),
                ("J2", "FLOW"): SimpleNamespace(
                    time_series="TS2", scale_factor=1.0,
                    base_value=0.0, pattern=None, mass_unit_factor=1.0,
                ),
            },
            "TIMESERIES": {
                "TS1": SimpleNamespace(data=[(0.0, 10.0), (1.0, 2.0)]),
                "TS2": SimpleNamespace(data=[(0.0, 4.0), (1.0, 1.0)]),
            },
        }

    def __getitem__(self, key):
        return self.data[key]


def test_base_scenario_preserves_each_nodes_effective_curve_and_scales_component():
    scenario = batch._base_inflow_scenario(_Inp(), factor=2.0, scenario_id="base_f2")

    assert scenario.node_series["J1"] == [(0.0, 45.0), (1.0, 13.0)]
    assert scenario.node_series["J2"] == [(0.0, 8.0), (1.0, 2.0)]
    assert scenario.last_time_hours == 1.0


def test_base_scenario_rejects_shared_timeseries_with_different_inflow_transform():
    inp = _Inp()
    inp.data["INFLOWS"][("J2", "FLOW")].time_series = "TS1"

    with pytest.raises(ValueError, match="shared.*cannot be rewritten independently"):
        batch._base_inflow_scenario(inp, factor=1.0, scenario_id="base_f1")


def test_base_scenario_preserves_baseline_before_late_series_start():
    inp = _Inp()
    inp.data["TIMESERIES"]["TS1"].data = [(0.5, 10.0), (1.0, 2.0)]

    scenario = batch._base_inflow_scenario(inp, factor=2.0, scenario_id="base_f2")

    assert scenario.node_series["J1"] == [(0.0, 5.0), (0.5, 45.0), (1.0, 13.0)]


def test_base_batch_retries_with_extended_drain_and_persists_state(monkeypatch, tmp_path):
    class _Config:
        network = SimpleNamespace(inp_path=tmp_path / "source.inp")
        validation = SimpleNamespace(
            drainage_extension_max_retries=2,
            drainage_extension_increment_hours=2.0,
            drain_down_hours=6.0,
            drainage_tail_window_hours=1.0,
            drainage_inflow_tolerance_lps=0.01,
            drainage_flooding_tolerance_lps=0.01,
            drainage_storage_change_tolerance_m3=0.01,
            inflow_closure_tolerance_lps=0.01,
        )

        @staticmethod
        def factors():
            return [1.0]

    class _Simulation:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            scenario_inp.with_suffix(".rpt").write_text("report")
            scenario_inp.with_suffix(".out").write_bytes(b"out")

        def __iter__(self):
            return iter(())

    scenario_inp = tmp_path / "base_f1.000.inp"
    scenario_inp.write_text("input")
    monkeypatch.setitem(sys.modules, "pyswmm", SimpleNamespace(Simulation=_Simulation))
    prepared_hours = []
    monkeypatch.setattr(
        batch,
        "_prepare_base_scenario_inp",
        lambda _config, _factor, _run_dir, hours: (prepared_hours.append(hours) or (scenario_inp, {})),
    )
    state_calls = iter(["not_drained", "drained"])
    monkeypatch.setattr(batch, "read_tail_window_observables", lambda *_args: {})
    monkeypatch.setattr(
        batch,
        "classify_drainage_state",
        lambda *_args, **_kwargs: (next(state_calls), {"state": "drained"}),
    )
    saved = []
    monkeypatch.setattr(
        batch,
        "save_drainage_state_manifest",
        lambda manifest, directory, name: saved.append((manifest, directory, name)),
    )

    result = batch.run_batch(_Config(), tmp_path / "runs")

    assert prepared_hours == [6.0, 8.0]
    assert result == [(1.0, scenario_inp.with_suffix(".rpt"))]
    assert saved[0][0]["extension_log"] == [
        {"attempt": 0, "drain_down_hours": 6.0, "state": "not_drained"},
        {"attempt": 1, "drain_down_hours": 8.0, "state": "drained"},
    ]
