from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.figures_part2.make_chico_norte_transfer_figures import (
    select_representative_scenarios,
    verify_sha256,
)


def test_representative_scenarios_use_positive_eligible_p10_p50_p90():
    totals = pd.DataFrame(
        [
            {"scenario_key": "dry", "total_swmm_m3": 0.0, "eligible": True},
            {"scenario_key": "excluded", "total_swmm_m3": 500.0, "eligible": False},
            {"scenario_key": "s1", "total_swmm_m3": 10.0, "eligible": True},
            {"scenario_key": "s2", "total_swmm_m3": 20.0, "eligible": True},
            {"scenario_key": "s3", "total_swmm_m3": 30.0, "eligible": True},
            {"scenario_key": "s4", "total_swmm_m3": 40.0, "eligible": True},
            {"scenario_key": "s5", "total_swmm_m3": 50.0, "eligible": True},
        ]
    )

    selected = select_representative_scenarios(totals)

    assert [item["label"] for item in selected] == ["bajo", "mediano", "alto"]
    assert [item["scenario_key"] for item in selected] == ["s1", "s3", "s5"]
    assert [item["quantile"] for item in selected] == [0.1, 0.5, 0.9]
    assert [item["total_swmm_m3"] for item in selected] == [10.0, 30.0, 50.0]


def test_representative_scenario_ties_break_by_scenario_key():
    totals = pd.DataFrame(
        [
            {"scenario_key": "z", "total_swmm_m3": 10.0, "eligible": True},
            {"scenario_key": "b", "total_swmm_m3": 20.0, "eligible": True},
            {"scenario_key": "a", "total_swmm_m3": 20.0, "eligible": True},
            {"scenario_key": "x", "total_swmm_m3": 30.0, "eligible": True},
        ]
    )

    selected = select_representative_scenarios(totals)

    assert selected[1]["scenario_key"] == "a"


def test_representative_scenarios_require_at_least_one_positive_eligible_run():
    totals = pd.DataFrame(
        [{"scenario_key": "dry", "total_swmm_m3": 0.0, "eligible": True}]
    )

    with pytest.raises(ValueError, match="positive eligible"):
        select_representative_scenarios(totals)


def test_sha256_verifier_accepts_exact_content_and_rejects_mismatch():
    path = Path(__file__)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    verify_sha256(path, expected, "test file")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(path, "0" * 64, "test file")


def test_prediction_frame_requires_exact_keys_and_registered_cascade():
    from scripts.figures_part2.make_chico_norte_transfer_figures import (
        KEY_COLUMNS,
        validate_prediction_frame,
    )

    key = {
        "run_id": 1,
        "network_id": 1,
        "scenario_id": 1,
        "scenario_key": "event-1",
        "scenario_kind": "hydrograph",
        "node_id": "N1",
        "factor_mult": 1.0,
        "shape_id": "shape-1",
    }
    predictions = pd.DataFrame(
        [{**key, "y_prob_clf": 0.8, "y_pred_clf": 1, "volume_conditional_m3": 10.0, "volume_routed_m3": 10.0}]
    )
    inputs = pd.DataFrame([{**key, "elev_fondo": 100.0}], columns=[*KEY_COLUMNS, "elev_fondo"])

    validate_prediction_frame(predictions, inputs, classification_threshold=0.5)

    with pytest.raises(ValueError, match="exact unique run-node coverage"):
        validate_prediction_frame(predictions, inputs.iloc[0:0], classification_threshold=0.5)

    wrong_route = predictions.copy()
    wrong_route.loc[0, "volume_routed_m3"] = 0.0
    with pytest.raises(ValueError, match="cascade"):
        validate_prediction_frame(wrong_route, inputs, classification_threshold=0.5)


def test_run_manifest_and_score_must_agree_on_eligible_ids():
    from scripts.figures_part2.make_chico_norte_transfer_figures import (
        validate_run_eligibility,
    )

    run_order = [
        {"run_id": 1, "eligible": True, "reason_codes": []},
        {"run_id": 2, "eligible": False, "reason_codes": ["drain_not_achieved"]},
    ]
    run_status = [
        {"run_id": 1, "eligible": True, "reason_codes": []},
        {"run_id": 2, "eligible": False, "reason_codes": ["drain_not_achieved"]},
    ]
    score = {
        "eligible_run_ids": [1],
        "label_sources": [
            {"run_id": 1, "swmm_rpt_sha256": "a" * 64, "scenario_key": "event-1"}
        ],
    }

    assert validate_run_eligibility(run_order, run_status, score) == {1}

    score["eligible_run_ids"] = [1, 2]
    with pytest.raises(ValueError, match="eligible run IDs"):
        validate_run_eligibility(run_order, run_status, score)


def test_registration_schema_allows_verified_staging_usage_marker():
    from scripts.figures_part2.make_chico_norte_transfer_figures import (
        validate_registration_schema,
    )

    validate_registration_schema(
        {"schema": "frozen_transfer_v1", "usage": "synthetic_only_unbound_sources"}
    )

    with pytest.raises(ValueError, match="unsupported schema"):
        validate_registration_schema({"schema": "unknown", "usage": "other"})


def test_expected_nodes_are_bound_to_each_eligible_run():
    from scripts.figures_part2.make_chico_norte_transfer_figures import (
        expected_nodes_by_run,
    )

    records = [
        {"run_id": 1, "expected_node_ids": ["N1", "N2"]},
        {"run_id": 2, "expected_node_ids": ["N3"]},
        {"run_id": 3, "expected_node_ids": ["N4"]},
    ]

    assert expected_nodes_by_run(records, {1, 3}) == {1: {"N1", "N2"}, 3: {"N4"}}

    with pytest.raises(ValueError, match="duplicate or invalid"):
        expected_nodes_by_run([records[0], records[0]], {1})


def test_volume_ticks_use_readable_log_decades_without_irregular_endpoint():
    from scripts.figures_part2.make_chico_norte_transfer_figures import _volume_ticks

    _positions, labels = _volume_ticks(115_988.0)

    assert labels == ["0", "1", "10", "100", "1000", "10000", "100000"]


def test_recomputed_full_score_metrics_must_match_frozen_score():
    from scripts.figures_part2.make_chico_norte_transfer_figures import (
        validate_recomputed_score,
    )

    frame = pd.DataFrame(
        {
            "vol_inundacion_m3": [0.0, 20.0],
            "inunda": [0, 1],
            "y_pred_clf_xgb": [0, 1],
            "volume_routed_m3_xgb": [0.0, 18.0],
        }
    )
    score = {
        "n_samples": 2,
        "n_actual_flooded": 1,
        "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 1},
        "precision": 1.0,
        "recall": 1.0,
        "rmse_m3": 2**0.5,
        "mae_m3": 1.0,
        "volume_bias_m3": -2.0,
    }

    validate_recomputed_score(frame, score, "xgboost")

    score["rmse_m3"] = 99.0
    with pytest.raises(ValueError, match="[Rr]ecomputed figure metrics"):
        validate_recomputed_score(frame, score, "xgboost")
