from swmm_resilience.ml.contracts import TABULAR_V4_21


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
