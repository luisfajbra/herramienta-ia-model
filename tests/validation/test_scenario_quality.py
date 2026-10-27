from __future__ import annotations

import pandas as pd

from swmm_resilience.validation.scenario_quality import (
    assess_scenario_quality,
    filter_simulation_results_by_quality,
)


_INITIAL = {"policy_id": "source_inp_initial_state_v1", "state_sha256": "a" * 64}
_REPORT = {
    "parser": "strict_text_v1",
    "parser_version": "strict_text_v1",
    "report_sha256": "b" * 64,
}


def _frames(nodes=("J1", "J2")):
    dynamic = pd.DataFrame({"node_id": list(nodes), "shape_id": "base", "factor_mult": 1.0})
    labels = pd.DataFrame({
        "node_id": list(nodes),
        "vol_inundacion_m3": [0.0] * len(nodes),
        "inunda": [0] * len(nodes),
    })
    labels.attrs["flooding_report_manifest"] = _REPORT
    return dynamic, labels


def _assessment(dynamic, labels, *, drainage="drained", continuity=1.0):
    return assess_scenario_quality(
        dynamic,
        labels,
        expected_node_ids=["J1", "J2"],
        continuity_error_pct=continuity,
        max_continuity_error_pct=5.0,
        continuity_limit_rationale="Provisional owner-approved 5% limit; hydraulic justification pending.",
        drainage_manifest={"state": drainage},
        initial_condition_manifest=_INITIAL,
    )


def test_scenario_quality_requires_complete_node_and_finite_label_coverage():
    dynamic, labels = _frames()
    assert _assessment(dynamic, labels)["eligible"] is True

    partial_dynamic, partial_labels = _frames(nodes=("J1",))
    result = _assessment(partial_dynamic, partial_labels)
    assert result["eligible"] is False
    assert "finite_coverage_failed" in result["reason_codes"]

    dynamic, labels = _frames()
    labels.loc[0, "vol_inundacion_m3"] = float("inf")
    assert "finite_coverage_failed" in _assessment(dynamic, labels)["reason_codes"]


def test_scenario_quality_rejects_bad_continuity_drainage_and_missing_report():
    dynamic, labels = _frames()
    result = _assessment(dynamic, labels, drainage="not_drained", continuity=6.0)
    assert result["eligible"] is False
    assert result["reason_codes"] == ["continuity_limit_exceeded", "drainage_not_drained"]

    labels.attrs.clear()
    result = _assessment(dynamic, labels)
    assert "flooding_report_provenance_missing" in result["reason_codes"]


def test_filter_removes_and_reports_only_ineligible_scenarios():
    eligible_dynamic, eligible_labels = _frames()
    eligible_dynamic["shape_id"] = "base"
    excluded_dynamic, excluded_labels = _frames()
    excluded_dynamic["shape_id"] = "storm_bad"
    scenarios = [
        (1.0, eligible_dynamic, eligible_labels),
        (1.0, excluded_dynamic, excluded_labels),
    ]
    decisions = [
        {"shape_id": "base", "factor_mult": 1.0, "eligible": True, "reason_codes": []},
        {"shape_id": "storm_bad", "factor_mult": 1.0, "eligible": False,
         "reason_codes": ["drainage_not_drained"]},
    ]

    kept, excluded = filter_simulation_results_by_quality(scenarios, decisions)

    assert len(kept) == 1
    assert kept[0][1]["shape_id"].iloc[0] == "base"
    assert excluded == [decisions[1]]


def test_filter_fails_closed_and_reports_result_without_assessment():
    dynamic, labels = _frames()
    result = (2.0, dynamic, labels)

    kept, excluded = filter_simulation_results_by_quality([result], [])

    assert kept == []
    assert excluded == [{
        "shape_id": "base",
        "factor_mult": 2.0,
        "eligible": False,
        "reason_codes": ["quality_assessment_missing"],
    }]
