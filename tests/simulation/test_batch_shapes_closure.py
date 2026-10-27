"""Tests for the inflow-closure wiring inside swmm_resilience.simulation.batch."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swmm_resilience.simulation import batch as batch_mod
from swmm_resilience.simulation import initial_conditions as initial_conditions_mod
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


@pytest.fixture(autouse=True)
def _stub_initial_condition_manifest(monkeypatch):
    calls = []

    def _save(inp_path, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source": Path(inp_path).name, "policy_id": "fixture-policy"}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        calls.append((Path(inp_path), output_path))
        return payload

    monkeypatch.setattr(
        initial_conditions_mod, "save_initial_condition_manifest", _save
    )
    return calls


def test_prepare_closed_scenario_inp_closes_a_shape_that_does_not_end_at_zero(
    monkeypatch, tmp_path, _stub_initial_condition_manifest,
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
    initial_path = tmp_path / "run" / "storm_x" / "storm_x_f1.000.initial_condition.json"
    assert json.loads(initial_path.read_text(encoding="utf-8")) == {
        "source": "base.inp",
        "policy_id": "fixture-policy",
    }
    assert _stub_initial_condition_manifest == [(config.network.inp_path, initial_path)]


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


def test_run_batch_shapes_classifies_and_saves_drainage_state(monkeypatch, tmp_path):
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    calls = {}

    def _fake_read(out_path, drain_down_hours, tail_window_hours):
        calls["read_args"] = (out_path, drain_down_hours, tail_window_hours)
        return {"n_report_steps": 3, "has_link_data": True}

    def _fake_classify(observables, **kwargs):
        calls["classify_kwargs"] = kwargs
        return "drained", {"policy_version": "tail_window_v1", "state": "drained", "failed_checks": []}

    class _FakeSimulation:
        def __init__(self, inp_path):
            self._inp_path = Path(inp_path)

        def __enter__(self):
            self._inp_path.with_suffix(".rpt").touch()
            self._inp_path.with_suffix(".out").touch()
            return iter([])

        def __exit__(self, *exc):
            return False

    import pyswmm
    monkeypatch.setattr(pyswmm, "Simulation", _FakeSimulation)
    monkeypatch.setattr(batch_mod, "read_tail_window_observables", _fake_read)
    monkeypatch.setattr(batch_mod, "classify_drainage_state", _fake_classify)

    config = SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.02,
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=2,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]
    shapes = {"storm_z": shape}

    batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    expected_out_path = tmp_path / "run" / "storm_z" / "storm_z_f1.000.out"
    assert calls["read_args"][0] == expected_out_path
    assert calls["read_args"][1] == 6.0  # drain_down_hours
    assert calls["read_args"][2] == 1.0  # drainage_tail_window_hours
    assert calls["classify_kwargs"]["inflow_tolerance_lps"] == 0.01
    assert calls["classify_kwargs"]["flooding_tolerance_lps"] == 0.02
    assert calls["classify_kwargs"]["storage_change_tolerance_m3"] == 0.03
    # residual_flow_tolerance_lps is documented to intentionally reuse
    # drainage_inflow_tolerance_lps (0.01), not a distinct config field --
    # this must equal 0.01 (drainage_inflow_tolerance_lps), NOT 0.02 or 0.03.
    assert calls["classify_kwargs"]["residual_flow_tolerance_lps"] == 0.01

    assert calls["classify_kwargs"]["window_hours"] == 1.0  # min(1.0, 6.0)

    manifest_path = tmp_path / "run" / "storm_z" / "storm_z_f1.000.drainage_state.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["state"] == "drained"
    # Fase C2 / revision final (I3): el spec exige registrar el incremento y el
    # limite CONFIGURADOS, no solo los intentos. En el caso comun (drena en el
    # intento 0) extension_log tiene una sola entrada, asi que sin estos dos
    # campos ninguno de los dos valores seria recuperable del manifiesto.
    assert payload["drainage_extension_increment_hours"] == pytest.approx(2.0)
    assert payload["drainage_extension_max_retries"] == 2
    assert len(payload["extension_log"]) == 1


def test_run_batch_shapes_does_not_warn_when_drainage_state_is_drained(monkeypatch, tmp_path):
    """Finding 6: a "drained" verdict must not trigger the not-drained warning."""
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    def _fake_read(out_path, drain_down_hours, tail_window_hours):
        return {"n_report_steps": 3, "has_link_data": True}

    def _fake_classify(observables, **kwargs):
        return "drained", {"policy_version": "tail_window_v1", "state": "drained", "failed_checks": []}

    class _FakeSimulation:
        def __init__(self, inp_path):
            self._inp_path = Path(inp_path)

        def __enter__(self):
            self._inp_path.with_suffix(".rpt").touch()
            self._inp_path.with_suffix(".out").touch()
            return iter([])

        def __exit__(self, *exc):
            return False

    import pyswmm
    monkeypatch.setattr(pyswmm, "Simulation", _FakeSimulation)
    monkeypatch.setattr(batch_mod, "read_tail_window_observables", _fake_read)
    monkeypatch.setattr(batch_mod, "classify_drainage_state", _fake_classify)

    config = SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.02,
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=2,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]  # ya termina en 0 -> sin warning de cierre
    shapes = {"storm_drained": shape}

    import warnings as warnings_mod

    with warnings_mod.catch_warnings():
        warnings_mod.simplefilter("error")
        results = batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    assert len(results) == 1


def test_run_batch_shapes_warns_when_drainage_state_is_not_drained(monkeypatch, tmp_path):
    """Finding 6: the classified state must not be silently discarded -- a
    non-"drained" verdict must surface as a warning. Fase C2: tambien debe
    agotar los reintentos configurados antes de rendirse."""
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    classify_calls = []

    def _fake_read(out_path, drain_down_hours, tail_window_hours):
        return {"n_report_steps": 3, "has_link_data": True}

    def _fake_classify(observables, **kwargs):
        classify_calls.append(kwargs)
        return "not_drained", {
            "policy_version": "tail_window_v1", "state": "not_drained",
            "failed_checks": ["inflow"],
        }

    class _FakeSimulation:
        def __init__(self, inp_path):
            self._inp_path = Path(inp_path)

        def __enter__(self):
            self._inp_path.with_suffix(".rpt").touch()
            self._inp_path.with_suffix(".out").touch()
            return iter([])

        def __exit__(self, *exc):
            return False

    import pyswmm
    monkeypatch.setattr(pyswmm, "Simulation", _FakeSimulation)
    monkeypatch.setattr(batch_mod, "read_tail_window_observables", _fake_read)
    monkeypatch.setattr(batch_mod, "classify_drainage_state", _fake_classify)

    config = SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.02,
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=2,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]  # ya termina en 0 -> sin warning de cierre
    shapes = {"storm_notdrained": shape}

    with pytest.warns(UserWarning, match="Drainage state: not_drained"):
        results = batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    assert len(results) == 1
    # max_retries=2 -> intentos 0, 1, 2 = 3 llamadas a classify_drainage_state
    assert len(classify_calls) == 3

    manifest_path = tmp_path / "run" / "storm_notdrained" / "storm_notdrained_f1.000.drainage_state.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["extension_log"]) == 3
    assert [entry["attempt"] for entry in payload["extension_log"]] == [0, 1, 2]
    assert payload["extension_log"][1]["drain_down_hours"] == pytest.approx(8.0)  # 6.0 + 1*2.0
    assert payload["extension_log"][2]["drain_down_hours"] == pytest.approx(10.0)  # 6.0 + 2*2.0
    assert payload["drainage_extension_increment_hours"] == pytest.approx(2.0)
    assert payload["drainage_extension_max_retries"] == 2


def test_run_batch_shapes_survives_an_unreadable_out_file(monkeypatch, tmp_path):
    """Finding 1: a broken/missing/truncated .out must not abort the whole
    batch -- the scenario degrades to "unverified" and the loop continues."""
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    def _raising_read(out_path, drain_down_hours, tail_window_hours):
        raise OSError("simulated truncated .out")

    class _FakeSimulation:
        def __init__(self, inp_path):
            self._inp_path = Path(inp_path)

        def __enter__(self):
            self._inp_path.with_suffix(".rpt").touch()
            self._inp_path.with_suffix(".out").touch()
            return iter([])

        def __exit__(self, *exc):
            return False

    import pyswmm
    monkeypatch.setattr(pyswmm, "Simulation", _FakeSimulation)
    monkeypatch.setattr(batch_mod, "read_tail_window_observables", _raising_read)

    config = SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.02,
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=2,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]  # ya termina en 0 -> sin warning de cierre
    shapes = {"storm_broken": shape}

    with pytest.warns(UserWarning, match="Could not read/classify the .out"):
        results = batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    # (a) run_batch_shapes does NOT raise, and (b) the scenario still
    # appears in the returned results.
    assert len(results) == 1
    assert results[0][0] == "storm_broken"

    # (c) a manifest is still written with state == "unverified" and
    # "out_read_error" in failed_checks.
    manifest_path = tmp_path / "run" / "storm_broken" / "storm_broken_f1.000.drainage_state.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["state"] == "unverified"
    assert "out_read_error" in payload["failed_checks"]


def test_run_batch_shapes_retries_until_drained(monkeypatch, tmp_path):
    """Fase C2: si el primer intento no drena pero uno posterior si, el
    bucle debe parar ahi -- no agotar los reintentos innecesariamente."""
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    states = iter(["not_drained", "drained"])

    def _fake_read(out_path, drain_down_hours, tail_window_hours):
        return {"n_report_steps": 3, "has_link_data": True}

    def _fake_classify(observables, **kwargs):
        state = next(states)
        return state, {
            "policy_version": "tail_window_v1", "state": state,
            "failed_checks": [] if state == "drained" else ["inflow"],
        }

    class _FakeSimulation:
        def __init__(self, inp_path):
            self._inp_path = Path(inp_path)

        def __enter__(self):
            self._inp_path.with_suffix(".rpt").touch()
            self._inp_path.with_suffix(".out").touch()
            return iter([])

        def __exit__(self, *exc):
            return False

    import pyswmm
    monkeypatch.setattr(pyswmm, "Simulation", _FakeSimulation)
    monkeypatch.setattr(batch_mod, "read_tail_window_observables", _fake_read)
    monkeypatch.setattr(batch_mod, "classify_drainage_state", _fake_classify)

    config = SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.02,
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=2,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]
    shapes = {"storm_retry_ok": shape}

    results = batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    assert len(results) == 1
    manifest_path = tmp_path / "run" / "storm_retry_ok" / "storm_retry_ok_f1.000.drainage_state.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["state"] == "drained"
    assert len(payload["extension_log"]) == 2  # paro en el intento 1 (drained), no llega a 2
    assert payload["extension_log"][-1]["state"] == "drained"


def test_run_batch_shapes_never_exceeds_max_retries(monkeypatch, tmp_path):
    """Fase C2: con max_retries=0 no debe haber ningun reintento -- un solo
    intento, igual que el comportamiento pre-C2."""
    import swmm_resilience.simulation.batch as batch_mod

    fake_inp = _FakeInp()
    monkeypatch.setattr(ts_mod, "load_inp", lambda _: fake_inp)

    classify_calls = []

    def _fake_read(out_path, drain_down_hours, tail_window_hours):
        return {"n_report_steps": 3, "has_link_data": True}

    def _fake_classify(observables, **kwargs):
        classify_calls.append(kwargs)
        return "not_drained", {
            "policy_version": "tail_window_v1", "state": "not_drained",
            "failed_checks": ["inflow"],
        }

    class _FakeSimulation:
        def __init__(self, inp_path):
            self._inp_path = Path(inp_path)

        def __enter__(self):
            self._inp_path.with_suffix(".rpt").touch()
            self._inp_path.with_suffix(".out").touch()
            return iter([])

        def __exit__(self, *exc):
            return False

    import pyswmm
    monkeypatch.setattr(pyswmm, "Simulation", _FakeSimulation)
    monkeypatch.setattr(batch_mod, "read_tail_window_observables", _fake_read)
    monkeypatch.setattr(batch_mod, "classify_drainage_state", _fake_classify)

    config = SimpleNamespace(
        validation=SimpleNamespace(
            inflow_closure_tolerance_lps=0.01, drain_down_hours=6.0,
            drainage_inflow_tolerance_lps=0.01, drainage_flooding_tolerance_lps=0.02,
            drainage_storage_change_tolerance_m3=0.03, drainage_tail_window_hours=1.0,
            drainage_extension_increment_hours=2.0, drainage_extension_max_retries=0,
        ),
        network=SimpleNamespace(inp_path=tmp_path / "base.inp"),
        factors=lambda: [1.0],
    )
    shape = [(0.0, 0.0), (0.05, 1.0), (0.1, 0.0)]
    shapes = {"storm_no_retry": shape}

    with pytest.warns(UserWarning, match="Drainage state: not_drained"):
        batch_mod.run_batch_shapes(config, shapes, {"1C": 2.0}, tmp_path / "run")

    assert len(classify_calls) == 1
