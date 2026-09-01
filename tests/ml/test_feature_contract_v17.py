import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.contracts import (
    FEATURE_DEFINITIONS_V17,
    FEATURE_COLUMNS_V17,
    TARGET_DEFINITIONS_V17,
    TABULAR_V3_17,
    FeatureContractError,
)

EXPECTED = (
    "elev_fondo", "prof_max", "n_tuberias_in", "n_tuberias_out",
    "diam_max_in", "diam_max_out", "pendiente_max_in", "pendiente_out",
    "base_inflow_lps", "dist_outfall_m", "n_nodos_aguas_arriba",
    "q_pico_acum_base", "upstream_capacity_lps", "q_pico_nodo",
    "q_pico_acum_escalado", "duracion_horas", "tiempo_al_pico_h",
)


def valid_frame():
    return pd.DataFrame([{name: float(i + 1) for i, name in enumerate(EXPECTED)}])


def test_contract_has_exact_id_order_and_count():
    assert TABULAR_V3_17.contract_id == "tabular_v3_17"
    assert FEATURE_COLUMNS_V17 == EXPECTED
    assert TABULAR_V3_17.feature_count == 17
    assert TABULAR_V3_17.descriptor_sha256 == (
        "56af955cadb90dda63b79f48dcec18bccaabc8eb33bcce07f5bf1874dcfbca8a"
    )
    assert all(item.units and item.description for item in FEATURE_DEFINITIONS_V17)
    assert {item.name for item in FEATURE_DEFINITIONS_V17 if item.nullable} == {
        "diam_max_in", "diam_max_out", "pendiente_max_in", "pendiente_out",
        "dist_outfall_m", "upstream_capacity_lps",
    }
    assert [(item.name, item.task) for item in TARGET_DEFINITIONS_V17] == [
        ("inunda", "classification"),
        ("vol_inundacion_m3", "regression"),
    ]


@pytest.mark.parametrize("columns", [
    EXPECTED[:15],
    EXPECTED[:-1],
    EXPECTED + ("extra",),
    EXPECTED[::-1],
    EXPECTED[:-1] + ("renamed_feature",),
    EXPECTED[:-1] + (EXPECTED[-2],),
    EXPECTED + ("inunda",),
])
def test_contract_rejects_wrong_feature_sets(columns):
    frame = pd.DataFrame([[1.0] * len(columns)], columns=columns)
    with pytest.raises(FeatureContractError):
        TABULAR_V3_17.validate_frame(frame)


def test_contract_rejects_null_required_values():
    frame = valid_frame()
    frame.loc[0, "duracion_horas"] = float("nan")
    with pytest.raises(FeatureContractError, match="duracion_horas"):
        TABULAR_V3_17.validate_frame(frame)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_contract_rejects_positive_and_negative_infinity(value):
    frame = valid_frame()
    frame.loc[0, "duracion_horas"] = value
    with pytest.raises(FeatureContractError, match="duracion_horas"):
        TABULAR_V3_17.validate_frame(frame)


def test_contract_rejects_boolean_feature_columns():
    frame = valid_frame()
    frame["prof_max"] = True
    with pytest.raises(FeatureContractError, match="prof_max"):
        TABULAR_V3_17.validate_frame(frame)


def test_contract_rejects_empty_and_numeric_strings():
    with pytest.raises(FeatureContractError, match="empty"):
        TABULAR_V3_17.validate_frame(valid_frame().iloc[0:0])
    frame = valid_frame()
    frame["prof_max"] = frame["prof_max"].astype(str)
    with pytest.raises(FeatureContractError, match="Non-numeric"):
        TABULAR_V3_17.validate_frame(frame)


def test_contract_allows_only_physical_nullable_features():
    frame = valid_frame()
    frame.loc[0, "diam_max_in"] = float("nan")
    TABULAR_V3_17.validate_frame(frame)


def test_contract_allows_all_null_nullable_column_for_storage_validation():
    frame = valid_frame()
    frame["dist_outfall_m"] = None
    TABULAR_V3_17.validate_frame(frame)


def test_contract_rejects_text_even_in_nullable_feature():
    frame = valid_frame()
    frame["diam_max_in"] = frame["diam_max_in"].astype(object)
    frame.loc[0, "diam_max_in"] = "unknown"
    with pytest.raises(FeatureContractError, match="Non-numeric"):
        TABULAR_V3_17.validate_frame(frame)


def _valid_frame():
    return pd.DataFrame(
        {name: [1.0, 2.0] for name in TABULAR_V3_17.feature_names}
    )


def test_rejects_complex_dtype_even_with_zero_imaginary_component():
    frame = _valid_frame()
    frame["elev_fondo"] = pd.array(
        [1 + 0j, 2 + 0j], dtype=complex
    )
    with pytest.raises(FeatureContractError, match="complex"):
        TABULAR_V3_17.validate_frame(frame)


def test_rejects_complex_dtype_with_nonzero_imaginary_component():
    frame = _valid_frame()
    frame["elev_fondo"] = pd.array([1 + 1j, 2 + 0j], dtype=complex)
    with pytest.raises(FeatureContractError, match="complex"):
        TABULAR_V3_17.validate_frame(frame)
