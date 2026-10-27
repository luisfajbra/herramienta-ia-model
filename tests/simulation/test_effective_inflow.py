from __future__ import annotations

import pytest

from swmm_resilience.simulation.effective_inflow import build_effective_inflow


_ENGINE_PROBE_INP = """\
[TITLE]
Effective inflow integration fixture
[OPTIONS]
FLOW_UNITS LPS
INFILTRATION HORTON
FLOW_ROUTING DYNWAVE
START_DATE 01/01/2025
START_TIME 00:00:00
REPORT_START_DATE 01/01/2025
REPORT_START_TIME 00:00:00
END_DATE 01/01/2025
END_TIME 00:10:00
REPORT_STEP 00:01:00
WET_STEP 00:01:00
DRY_STEP 00:01:00
ROUTING_STEP 0:00:30
[JUNCTIONS]
J1 10 2 0 0 0
[OUTFALLS]
O1 0 FREE NO
[CONDUITS]
C1 J1 O1 100 0.013 0 0 0 0
[XSECTIONS]
C1 CIRCULAR 1 0 0 0 1
[INFLOWS]
J1 FLOW TS FLOW 1 2 5
[TIMESERIES]
TS 0:00 10
TS 0:10 10
"""


def test_effective_flow_applies_sfactor_and_baseline_in_lps():
    inflow = build_effective_inflow(
        [(0.0, 10.0), (1.0, 10.0)],
        flow_units="LPS",
        constituent="FLOW",
        scale_factor=2.0,
        baseline=5.0,
    )

    assert inflow.at(0.5) == 25.0
    assert inflow.points == ((0.0, 25.0), (1.0, 25.0))
    assert inflow.metadata["source_timeseries"] is None
    assert inflow.metadata["mass_unit_factor"] == 1.0
    assert inflow.metadata["mass_unit_factor_applied"] is False


@pytest.mark.parametrize("flow_units", [None, "CFS"])
def test_effective_flow_rejects_missing_or_non_lps_units(flow_units):
    with pytest.raises(ValueError, match="FLOW_UNITS.*LPS"):
        build_effective_inflow(
            [(0.0, 10.0), (1.0, 10.0)],
            flow_units=flow_units,
            constituent="FLOW",
        )


def test_effective_flow_rejects_unsupported_constituent_pattern_and_non_finite_values():
    base = [(0.0, 10.0), (1.0, 12.0)]
    with pytest.raises(ValueError, match="constituent"):
        build_effective_inflow(base, flow_units="LPS", constituent="TSS")
    with pytest.raises(ValueError, match="pattern"):
        build_effective_inflow(base, flow_units="LPS", constituent="FLOW", pattern="DAILY")
    with pytest.raises(ValueError, match="finite"):
        build_effective_inflow(
            [(0.0, 10.0), (1.0, float("inf"))],
            flow_units="LPS",
            constituent="FLOW",
        )


def test_effective_flow_uses_zero_time_series_outside_its_range_and_preserves_baseline():
    inflow = build_effective_inflow(
        [(1.0, 10.0), (2.0, 20.0)],
        flow_units="LPS",
        constituent="FLOW",
        scale_factor=2.0,
        baseline=5.0,
    )

    assert inflow.at(0.0) == 5.0
    assert inflow.at(1.5) == 35.0
    assert inflow.at(2.5) == 5.0


def test_effective_flow_keeps_node_series_independent():
    node_a = build_effective_inflow(
        [(0.0, 10.0), (1.0, 20.0)], flow_units="LPS", constituent="FLOW"
    )
    node_b = build_effective_inflow(
        [(0.0, 3.0), (1.0, 4.0)], flow_units="LPS", constituent="FLOW"
    )

    assert node_a.at(0.5) == 15.0
    assert node_b.at(0.5) == 3.5


def test_timeseries_multiplier_scales_series_component_but_preserves_baseline():
    inflow = build_effective_inflow(
        [(0.0, 10.0), (1.0, 10.0)],
        flow_units="LPS",
        constituent="FLOW",
        scale_factor=2.0,
        baseline=5.0,
    )

    assert inflow.at_scaled_component(0.5, 2.0) == 45.0
    assert inflow.at(0.5) == 25.0


def test_extraction_and_swmm_engine_agree_on_effective_flow_at_start(tmp_path):
    from pyswmm import Nodes, Simulation

    from swmm_resilience.extraction.static_features import _get_peak_inflows
    from swmm_resilience.simulation.runner import _profile_value_lps
    from swmm_resilience.simulation.swmm_api_io import get_node_inflow_profiles, load_inp

    inp_path = tmp_path / "effective_flow_engine.inp"
    inp_path.write_text(_ENGINE_PROBE_INP, encoding="utf-8")
    inp = load_inp(inp_path)
    profiles = get_node_inflow_profiles(inp)

    with Simulation(str(inp_path)) as simulation:
        node = Nodes(simulation)["J1"]
        next(iter(simulation))
        engine_lps = node.lateral_inflow

    assert _profile_value_lps(profiles["J1"], 0.0) == pytest.approx(engine_lps)
    assert _get_peak_inflows(inp)["J1"] == pytest.approx(25.0)
