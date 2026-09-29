import pandas as pd
import pytest

from swmm_resilience.ml.contracts import FeatureContractError, TABULAR_V4_21


def test_tabular_v4_21_has_21_ordered_features():
    assert TABULAR_V4_21.contract_id == "tabular_v4_21"
    assert TABULAR_V4_21.feature_count == 21


def test_tabular_v4_21_drops_duracion_and_tiempo_al_pico():
    names = TABULAR_V4_21.feature_names
    assert "duracion_horas" not in names
    assert "tiempo_al_pico_h" not in names


def test_tabular_v4_21_keeps_q_pico_acum_escalado_as_a_proxy():
    assert "q_pico_acum_escalado" in TABULAR_V4_21.feature_names


def test_tabular_v4_21_includes_the_six_new_tributary_features():
    names = set(TABULAR_V4_21.feature_names)
    assert names.issuperset({
        "volumen_local_integrado_m3", "volumen_tributario_integrado_m3",
        "duracion_efectiva_tributario_h", "tiempo_primer_pico_tributario_h",
        "tiempo_sobre_capacidad_h", "integral_exceso_capacidad_m3",
    })


def test_tabular_v4_21_marks_the_capacity_dependent_features_nullable():
    from swmm_resilience.ml.contracts import NULLABLE_FEATURE_COLUMNS_V4_21
    assert "tiempo_sobre_capacidad_h" in NULLABLE_FEATURE_COLUMNS_V4_21
    assert "integral_exceso_capacidad_m3" in NULLABLE_FEATURE_COLUMNS_V4_21


def test_tabular_v3_17_is_unchanged():
    """No-regresion: este plan nunca toca el contrato legado."""
    from swmm_resilience.ml.contracts import TABULAR_V3_17
    assert TABULAR_V3_17.contract_id == "tabular_v3_17"
    assert TABULAR_V3_17.feature_count == 17


def _valid_v4_21_frame():
    """Frame `tabular_v4_21` bien formado: 1.0 en cada columna requerida,
    NaN en las dos columnas capacity-dependent (el resultado real de
    compute_node_tributary_features para un nodo con capacidad desconocida)."""
    frame = pd.DataFrame(
        [{name: 1.0 for name in TABULAR_V4_21.feature_names}]
    )
    frame.loc[0, "tiempo_sobre_capacidad_h"] = float("nan")
    frame.loc[0, "integral_exceso_capacidad_m3"] = None
    return frame


def test_tabular_v4_21_validate_frame_accepts_nulls_in_its_own_nullable_columns():
    """Regresion del hallazgo 1: validate_frame() hardcodeaba
    NULLABLE_FEATURE_COLUMNS_V17 en vez de derivar la nulabilidad de
    self.definitions, asi que el nullable=True de tabular_v4_21 en
    tiempo_sobre_capacidad_h/integral_exceso_capacidad_m3 no tenia efecto y
    el contrato rechazaba su propia salida correcta."""
    frame = _valid_v4_21_frame()
    TABULAR_V4_21.validate_frame(frame)  # no debe lanzar


def test_tabular_v4_21_validate_frame_still_rejects_null_in_a_non_nullable_column():
    """El fix no debe desactivar la validacion por completo: una columna que
    NO es nullable en tabular_v4_21 (p.ej. volumen_local_integrado_m3) sigue
    siendo obligatoria."""
    frame = _valid_v4_21_frame()
    frame.loc[0, "volumen_local_integrado_m3"] = float("nan")
    with pytest.raises(FeatureContractError, match="volumen_local_integrado_m3"):
        TABULAR_V4_21.validate_frame(frame)
