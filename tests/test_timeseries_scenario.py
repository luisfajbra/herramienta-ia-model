from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.simulation import timeseries_scenario as ts_mod


class _FakeTs:
    def __init__(self, data):
        self.data = list(data)


class _FakeInflow:
    def __init__(self, ts_name):
        self.time_series = ts_name


class _FakeInp:
    def __init__(self, shared_series=False):
        ts_name_2c = "1C" if shared_series else "2C"
        self._s = {
            "INFLOWS": {
                ("1C", "FLOW"): _FakeInflow("1C"),
                ("2C", "FLOW"): _FakeInflow(ts_name_2c),
            },
            "TIMESERIES": {
                "1C": _FakeTs([(0.0, 0.6), (0.05, 0.973)]),
                "2C": _FakeTs([(0.0, 0.825), (0.05, 1.337)]),
            },
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


def _make_scenario(last_time_hours=0.05):
    return HydrographScenario(
        scenario_id="test_scen",
        node_series={
            "1C": [(0.0, 1.0), (last_time_hours, 2.0)],
            "2C": [(0.0, 0.5), (last_time_hours, 1.5)],
        },
        time_grid_hours=[0.0, last_time_hours],
        last_time_hours=last_time_hours,
    )


def test_replaces_all_expected_series(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    scenario = _make_scenario()
    ts_mod.write_scenario_inp(tmp_path / "base.inp", scenario, tmp_path / "out")

    assert fake_inp["TIMESERIES"]["1C"].data == scenario.node_series["1C"]
    assert fake_inp["TIMESERIES"]["2C"].data == scenario.node_series["2C"]


def test_inflows_section_unchanged(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    original_inflows = dict(fake_inp["INFLOWS"])
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(), tmp_path / "out")

    assert fake_inp["INFLOWS"] == original_inflows


def test_preserves_series_names(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(), tmp_path / "out")

    assert "1C" in fake_inp["TIMESERIES"]
    assert "2C" in fake_inp["TIMESERIES"]


def test_base_inp_file_not_modified(monkeypatch, tmp_path):
    base = tmp_path / "base.inp"
    base.write_text("original content")
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(base, _make_scenario(), tmp_path / "out")

    assert base.read_text() == "original content"


def test_adjusts_end_time_no_date_rollover(monkeypatch, tmp_path):
    # last_time=5h, base_duration=3h → new end = 8h → same date
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(5.0), tmp_path / "out")

    assert fake_inp["OPTIONS"]["END_TIME"] == "08:00:00"
    assert fake_inp["OPTIONS"]["END_DATE"] == datetime.date(2025, 1, 4)


def test_adjusts_end_date_on_midnight_crossover(monkeypatch, tmp_path):
    # last_time=22h, base_duration=3h → new end = 25h → next day 01:00:00
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(22.0), tmp_path / "out")

    assert fake_inp["OPTIONS"]["END_TIME"] == "01:00:00"
    assert fake_inp["OPTIONS"]["END_DATE"] == datetime.date(2025, 1, 5)


def test_preserves_routing_and_report_step(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    fake_inp["OPTIONS"]["ROUTING_STEP"] = "0:00:05"
    fake_inp["OPTIONS"]["REPORT_STEP"] = "00:05:00"
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(), tmp_path / "out")

    assert fake_inp["OPTIONS"]["ROUTING_STEP"] == "0:00:05"
    assert fake_inp["OPTIONS"]["REPORT_STEP"] == "00:05:00"


def test_rejects_shared_series(monkeypatch, tmp_path):
    # Both 1C and 2C reference the same series "1C"
    fake_inp = _FakeInp(shared_series=True)
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    with pytest.raises(ValueError, match="compartida"):
        ts_mod.write_scenario_inp(tmp_path / "base.inp", _make_scenario(), tmp_path / "out")


def test_writes_file_to_output_dir(monkeypatch, tmp_path):
    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    result = ts_mod.write_scenario_inp(
        tmp_path / "base.inp", _make_scenario(), tmp_path / "out"
    )

    assert fake_inp.written_to is not None
    assert str(tmp_path / "out") in fake_inp.written_to
    assert result == Path(fake_inp.written_to)
