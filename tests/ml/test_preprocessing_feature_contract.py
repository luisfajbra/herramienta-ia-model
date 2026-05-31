import pandas as pd

from swmm_resilience.ml.preprocessing import get_feature_columns


def test_tabular_feature_selection_keeps_static_hydraulic_features():
    df = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "node_id": ["J1"],
            "scenario_type": ["steady"],
            "spatial_pattern": ["uniform"],
            "delta_inflow_lps": [0.0],
            "inflow_multiplier": [2.0],
            "invert_elev_m": [100.0],
            "full_depth_m": [2.0],
            "base_inflow_lps": [1.5],
            "in_degree": [2],
            "out_degree": [1],
            "upstream_capacity_lps": [50.0],
            "downstream_capacity_lps": [40.0],
            "flooded": [1],
            "peak_flooding_lps": [12.0],
            "flooding_duration_min": [5.0],
            "max_depth_m": [1.8],
            "max_depth_ratio": [0.9],
            "time_to_peak_min": [15.0],
            "depth_rate_m_per_min": [0.2],
            "max_total_outflow_lps": [30.0],
            "time_to_peak_outflow_min": [20.0],
        }
    )

    features = get_feature_columns(df, target="peak_flooding_lps")

    assert "inflow_multiplier" in features
    assert "invert_elev_m" in features
    assert "full_depth_m" in features
    assert "base_inflow_lps" in features
    assert "in_degree" in features
    assert "out_degree" in features
    assert "upstream_capacity_lps" in features
    assert "downstream_capacity_lps" in features


def test_tabular_feature_selection_drops_result_and_metadata_columns():
    df = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "node_id": ["J1"],
            "scenario_type": ["steady"],
            "spatial_pattern": ["uniform"],
            "delta_inflow_lps": [0.0],
            "inflow_multiplier": [2.0],
            "in_degree": [2],
            "out_degree": [1],
            "upstream_capacity_lps": [50.0],
            "downstream_capacity_lps": [40.0],
            "flooded": [1],
            "peak_flooding_lps": [12.0],
            "flooding_duration_min": [5.0],
            "max_depth_m": [1.8],
            "max_depth_ratio": [0.9],
            "time_to_peak_min": [15.0],
            "depth_rate_m_per_min": [0.2],
            "max_total_outflow_lps": [30.0],
            "time_to_peak_outflow_min": [20.0],
        }
    )

    features = get_feature_columns(df, target="peak_flooding_lps")

    assert set(features) == {
        "inflow_multiplier",
        "in_degree",
        "out_degree",
        "upstream_capacity_lps",
        "downstream_capacity_lps",
    }
    assert "run_id" not in features
    assert "node_id" not in features
    assert "delta_inflow_lps" not in features
    assert "flooded" not in features
    assert "peak_flooding_lps" not in features
    assert "flooding_duration_min" not in features
    assert "max_depth_m" not in features
    assert "max_depth_ratio" not in features
    assert "time_to_peak_min" not in features
    assert "depth_rate_m_per_min" not in features
    assert "max_total_outflow_lps" not in features
    assert "time_to_peak_outflow_min" not in features
