import pytest

from scripts.review_chico_norte_hydraulics import make_sensitivity_inp, parse_routing_diagnostics


def test_sensitivity_changes_only_requested_options_and_initial_link_flow():
    source = """[OPTIONS]
REPORT_STEP 0:03:00
ROUTING_STEP 0:00:05
MINIMUM_STEP 0.5
[CONDUITS]
;;Name From To Length Roughness InOffset OutOffset InitFlow MaxFlow
T1 N1 N2 10 0.013 0 0 42 0
T2 N2 N3 12 0.015 0 0 6.5 0
[JUNCTIONS]
N1 0 1 0 0 0
"""
    changed, count = make_sensitivity_inp(source, routing_seconds=1, zero_initial=True)
    assert count == 2
    assert "REPORT_STEP 00:01:00" in changed
    assert "ROUTING_STEP 00:00:01" in changed
    assert "MINIMUM_STEP 0.5" in changed
    assert "T1 N1 N2 10 0.013 0 0 0 0" in changed
    assert "T2 N2 N3 12 0.015 0 0 0 0" in changed
    assert "N1 0 1 0 0 0" in changed


def test_sensitivity_refuses_missing_routing_option():
    with pytest.raises(ValueError, match="ROUTING_STEP"):
        make_sensitivity_inp("[OPTIONS]\nREPORT_STEP 0:03:00\n[CONDUITS]\n", routing_seconds=1, zero_initial=False)


def test_parse_routing_diagnostics_reads_percent_and_named_instability():
    report = """Flow Routing Continuity
Continuity Error (%) ..... 0.328
Highest Flow Instability Indexes
********************************
Link T463 (112)
Link T508 (73)
Most Frequent Nonconverging Nodes
********************************
Node 523C (72.09%)
Routing Time Step Summary
*************************
Minimum Time Step : 0.50 sec
Average Time Step : 0.84 sec
Maximum Time Step : 5.00 sec
% of Steps Not Converging : 72.09
"""
    result = parse_routing_diagnostics(report)
    assert result["continuity_error_pct"] == 0.328
    assert result["steps_not_converging_pct"] == 72.09
    assert result["top_fii_link"] == "T463"
    assert result["top_fii"] == 112
    assert result["top_nonconverging_node"] == "523C"
    assert result["routing_average_s"] == 0.84


def test_flow_balance_converts_million_litres_to_cubic_metres():
    from scripts.inventory_chico_norte_hydraulic_reports import parse_flow_balance_m3

    report = """Flow Routing Continuity        hectare-m      10^6 ltr
External Inflow ..........        37.676       376.760
External Outflow .........        26.346       263.461
Flooding Loss ............        11.662       116.620
Initial Stored Volume ....         0.405         4.048
Final Stored Volume ......         0.000         0.000
Time-Step Critical Elements
"""
    balance = parse_flow_balance_m3(report)
    assert balance["aporte_externo_m3"] == 376760
    assert balance["volumen_inicial_m3"] == 4048
    assert balance["perdida_inundacion_m3"] == 116620


def test_verify_input_changes_rejects_unexpected_conduit_geometry():
    from scripts.inventory_chico_norte_hydraulic_reports import compare_inputs

    original = "[OPTIONS]\nROUTING_STEP 00:00:05\n[CONDUITS]\nT1 N1 N2 10 0.013 0 0 42 0\n"
    changed_flow = original.replace("0 0 42 0", "0 0 0 0")
    assert compare_inputs(original, changed_flow, {"init": True}) == {"InitFlow": 1}
    changed_geometry = changed_flow.replace("N2 10", "N2 11")
    with pytest.raises(ValueError, match="unexpected change"):
        compare_inputs(original, changed_geometry, {"init": True})


def test_native_step_windows_include_edges_but_exclude_unobserved_gaps():
    from scripts.review_chico_norte_native_steps import within_windows

    assert within_windows(0)
    assert within_windows(900)
    assert not within_windows(901)
    assert within_windows(40500)
    assert within_windows(45000)
    assert not within_windows(45001)


def test_irregular_native_series_compare_only_over_common_interval():
    from scripts.summarize_chico_norte_native_steps import compare_on_common_seconds

    result = compare_on_common_seconds([0.2, 2.2, 4.2], [0.2, 2.2, 4.2],
                                       [1.1, 3.1, 5.1], [2.1, 4.1, 6.1])
    assert result["start_s"] == 2
    assert result["end_s"] == 4
    assert result["max_abs_difference"] == pytest.approx(1.0)
    assert result["rmse_difference"] == pytest.approx(1.0)
