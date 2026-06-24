import pandas as pd
import pytest

from swmm_resilience.dataset.assembler import assemble_dataset
from swmm_resilience.dataset.validator import validate_dataset
from swmm_resilience.extraction.dynamic_features import compute_dynamic_features


def static_topology_df():
    return pd.DataFrame(
        {
            "node_id": ["J1", "J2"],
            "elev_fondo": [10.0, 9.5],
            "prof_max": [1.5, 1.4],
            "diam_max_in": [None, 0.6],
            "diam_max_out": [0.6, 0.7],
            "pendiente_max_in": [None, 0.01],
            "pendiente_out": [0.01, 0.02],
            "base_inflow_lps": [2.0, 3.0],
            "dist_outfall_m": [100.0, 40.0],
            "n_nodos_aguas_arriba": [0, 1],
            "q_pico_acum_base": [2.0, 5.0],
            "upstream_capacity_lps": [None, 25.0],
            "coord_x": [0.0, 1.0],
            "coord_y": [0.0, 1.0],
        }
    )


def test_dynamic_features_scale_peak_and_accumulated_flow():
    df = compute_dynamic_features(static_topology_df(), 1.5)

    assert df.to_dict("records") == [
        {"node_id": "J1", "factor_mult": 1.5, "q_pico_nodo": 3.0, "q_pico_acum_escalado": 3.0},
        {"node_id": "J2", "factor_mult": 1.5, "q_pico_nodo": 4.5, "q_pico_acum_escalado": 7.5},
    ]


def test_assemble_dataset_fills_missing_labels_as_zero(tmp_path):
    static_df = static_topology_df()
    dynamic_df = compute_dynamic_features(static_df, 1.0)
    labels_df = pd.DataFrame({"node_id": ["J2"], "vol_inundacion_m3": [12.0], "inunda": [1]})

    dataset = assemble_dataset(static_df, [(1.0, dynamic_df, labels_df)], tmp_path / "dataset.csv")

    assert len(dataset) == 2
    row_j1 = dataset[dataset["node_id"] == "J1"].iloc[0]
    assert row_j1["vol_inundacion_m3"] == 0.0
    assert row_j1["inunda"] == 0
    row_j2 = dataset[dataset["node_id"] == "J2"].iloc[0]
    assert row_j2["vol_inundacion_m3"] == 12.0
    assert row_j2["inunda"] == 1
    assert (tmp_path / "dataset.csv").exists()


def test_assemble_dataset_rejects_missing_label_columns(tmp_path):
    static_df = static_topology_df()
    dynamic_df = compute_dynamic_features(static_df, 1.0)
    labels_df = pd.DataFrame({"node_id": ["J2"], "vol_inundacion_m3": [12.0]})

    with pytest.raises(ValueError, match="labels_df debe incluir columnas vol_inundacion_m3 e inunda"):
        assemble_dataset(static_df, [(1.0, dynamic_df, labels_df)], tmp_path / "dataset.csv")


def test_validate_dataset_rejects_wrong_row_count():
    df = pd.DataFrame({"inunda": [0], "vol_inundacion_m3": [0.0]})

    with pytest.raises(ValueError, match="Filas esperadas"):
        validate_dataset(df, n_nodes=2, n_factors=1)
