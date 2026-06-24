import pytest

from swmm_resilience.visualization.labels import (
    feature_display_name,
    format_node_label,
)


@pytest.mark.parametrize(
    ("node_id", "expected"),
    [
        ("123I", "123"),
        ("123O", "123"),
        ("123C", "123"),
        ("123i", "123"),
        ("123o", "123"),
        ("123c", "123"),
        ("81Ib", "81"),
        ("81ib", "81"),
        ("J123", "J123"),
        ("IO-4", "IO-4"),
        ("12A", "12"),
        ("ABC", "ABC"),
    ],
)
def test_format_node_label_removes_alphabetic_suffix_from_numeric_id(
    node_id, expected
):
    assert format_node_label(node_id) == expected


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("elev_fondo", "Invert Elevation"),
        ("prof_max", "Maximum Depth"),
        ("diam_max_in", "Maximum Inlet Diameter"),
        ("diam_max_out", "Maximum Outlet Diameter"),
        ("pendiente_max_in", "Maximum Inlet Slope"),
        ("pendiente_out", "Outlet Slope"),
        ("base_inflow_lps", "Base Inflow"),
        ("dist_outfall_m", "Distance to Outfall"),
        ("n_nodos_aguas_arriba", "Upstream Node Count"),
        ("q_pico_acum_base", "Base Accumulated Peak Flow"),
        ("upstream_capacity_lps", "Upstream Capacity"),
        ("factor_mult", "Flow Multiplier"),
        ("q_pico_nodo", "Node Peak Inflow"),
        ("q_pico_acum_escalado", "Scaled Accumulated Peak Flow"),
    ],
)
def test_feature_display_name_explains_current_model_features(feature, expected):
    assert feature_display_name(feature) == expected


def test_feature_display_name_has_readable_fallback():
    assert feature_display_name("future_feature_name") == "Future Feature Name"
