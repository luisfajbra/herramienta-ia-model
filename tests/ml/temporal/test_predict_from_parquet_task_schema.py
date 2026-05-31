import pytest

from swmm_resilience.ml.temporal.predict import _output_columns_for_temporal_task


def test_classification_schema_uses_probability_columns():
    assert _output_columns_for_temporal_task("classification") == [
        "node_id",
        "max_flood_prob",
        "mean_flood_prob",
        "windows_total",
        "windows_flood_predicted",
        "actual_flooded",
    ]


def test_regression_schema_uses_peak_lps_columns():
    assert _output_columns_for_temporal_task("regression") == [
        "node_id",
        "max_peak_flooding_lps_pred",
        "mean_peak_flooding_lps_pred",
        "windows_total",
        "actual_peak_flooding_lps",
    ]


def test_unknown_temporal_task_is_rejected():
    with pytest.raises(ValueError, match="classification.*regression"):
        _output_columns_for_temporal_task("ranking")
