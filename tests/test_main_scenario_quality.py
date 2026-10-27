import main
import pandas as pd

from swmm_resilience.validation.scenario_quality import assess_scenario_quality


def test_failed_flood_report_is_recorded_as_failed_labels_not_zero_labels(tmp_path):
    labels = main._extract_labels_for_quality(
        tmp_path / "missing.rpt", ["J1", "J2"], threshold_m3=1.0
    )

    assert labels["node_id"].tolist() == ["J1", "J2"]
    assert labels["inunda"].tolist() == [0, 0]
    assert labels.attrs["flooding_report_manifest"]["failure_code"] == "missing_report"

    decision = assess_scenario_quality(
        pd.DataFrame({"node_id": ["J1", "J2"], "feature": [1.0, 2.0]}),
        labels,
        expected_node_ids=["J1", "J2"],
        continuity_error_pct=0.0,
        max_continuity_error_pct=5.0,
        continuity_limit_rationale="Fixture limit.",
        drainage_manifest={"state": "drained"},
        initial_condition_manifest={"policy_id": "fixture", "state_sha256": "a" * 64},
    )
    assert decision["eligible"] is False
    assert decision["reason_codes"] == ["flooding_report_invalid"]
