import json

import pytest

from swmm_resilience.simulation.inflow_closure import (
    INFLOW_CLOSURE_POLICY_VERSION,
    apply_closure_policy,
    close_series_to_zero,
    enforce_strict_zero_endpoint,
    save_closure_manifest,
)
from swmm_resilience.validation.hydrograph_csv import HydrographScenario


def test_close_series_to_zero_leaves_an_already_closed_series_untouched():
    series = [(0.0, 5.0), (1.0, 2.0), (2.0, 0.0)]
    closed, changed = close_series_to_zero(series, tolerance_lps=0.01)
    assert closed == series
    assert changed is False


def test_close_series_to_zero_appends_one_point_at_the_next_grid_step():
    series = [(0.0, 5.0), (1.0, 8.0), (2.0, 3.0)]
    closed, changed = close_series_to_zero(series, tolerance_lps=0.01)
    assert changed is True
    assert closed[:-1] == series
    assert closed[-1] == (3.0, 0.0)  # step = 2.0 - 1.0 = 1.0


def test_close_series_to_zero_uses_the_last_interval_as_the_step():
    series = [(0.0, 5.0), (1.5, 8.0), (2.0, 3.0)]  # irregular grid, last step = 0.5
    closed, _ = close_series_to_zero(series, tolerance_lps=0.01)
    assert closed[-1] == (2.5, 0.0)


def test_close_series_to_zero_treats_a_value_within_tolerance_as_already_closed():
    series = [(0.0, 5.0), (1.0, 0.005)]
    closed, changed = close_series_to_zero(series, tolerance_lps=0.01)
    assert changed is False
    assert closed == series


def test_close_series_to_zero_rejects_a_series_with_fewer_than_two_points():
    with pytest.raises(ValueError, match="2 puntos"):
        close_series_to_zero([(0.0, 5.0)], tolerance_lps=0.01)


def _scenario(node_series, time_grid_hours=None, last_time_hours=None):
    grid = time_grid_hours or sorted({t for s in node_series.values() for t, _ in s})
    last = last_time_hours if last_time_hours is not None else max(grid)
    return HydrographScenario(
        scenario_id="s1", node_series=node_series,
        time_grid_hours=grid, last_time_hours=last,
    )


def test_apply_closure_policy_closes_every_node_that_needs_it():
    scenario = _scenario({
        "A": [(0.0, 5.0), (1.0, 3.0)],   # needs closing
        "B": [(0.0, 5.0), (1.0, 0.0)],   # already closed
    })
    closed_scenario, manifest = apply_closure_policy(scenario, tolerance_lps=0.01)

    assert closed_scenario.node_series["A"][-1] == (2.0, 0.0)
    assert closed_scenario.node_series["B"] == scenario.node_series["B"]
    assert closed_scenario.last_time_hours == 2.0
    assert manifest["changed_nodes"] == ["A"]
    assert manifest["policy_version"] == INFLOW_CLOSURE_POLICY_VERSION
    assert manifest["tolerance_lps"] == 0.01
    assert manifest["source_scenario_id"] == "s1"
    assert manifest["original_last_time_hours"] == 1.0
    assert manifest["effective_last_time_hours"] == 2.0


def test_apply_closure_policy_is_a_no_op_when_nothing_needs_closing():
    scenario = _scenario({"A": [(0.0, 5.0), (1.0, 0.0)]})
    closed_scenario, manifest = apply_closure_policy(scenario, tolerance_lps=0.01)

    assert closed_scenario is scenario
    assert manifest["changed_nodes"] == []
    assert manifest["effective_last_time_hours"] == manifest["original_last_time_hours"]


def test_apply_closure_policy_never_mutates_the_input_scenario():
    original_series = [(0.0, 5.0), (1.0, 3.0)]
    scenario = _scenario({"A": list(original_series)})
    apply_closure_policy(scenario, tolerance_lps=0.01)
    assert scenario.node_series["A"] == original_series


def test_enforce_strict_zero_endpoint_accepts_an_already_closed_scenario():
    scenario = _scenario({"A": [(0.0, 5.0), (1.0, 0.0)]})
    enforce_strict_zero_endpoint(scenario, tolerance_lps=0.01)  # must not raise


def test_enforce_strict_zero_endpoint_rejects_and_never_modifies():
    original_series = [(0.0, 5.0), (1.0, 3.0)]
    scenario = _scenario({"A": list(original_series)})
    with pytest.raises(ValueError, match="A"):
        enforce_strict_zero_endpoint(scenario, tolerance_lps=0.01)
    assert scenario.node_series["A"] == original_series


def test_save_closure_manifest_writes_readable_json(tmp_path):
    manifest = {
        "policy_version": INFLOW_CLOSURE_POLICY_VERSION, "tolerance_lps": 0.01,
        "source_scenario_id": "s1", "changed_nodes": ["A"],
        "original_last_time_hours": 1.0, "effective_last_time_hours": 2.0,
    }
    path = save_closure_manifest(manifest, tmp_path, "s1")
    assert path.name == "s1.closure_manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
