import dataclasses

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.baselines import (
    BASELINES,
    FORMA_MAS_CERCANA,
    INTERPOLACION_FACTOR,
    BaselineNotApplicable,
    baseline_oof,
    evaluate_baseline,
)
from swmm_resilience.ml.bench.schemas import OOF_COLUMNS, PROTOCOL_INTERPRETATION
from tests.ml.bench.prepared_factory import make_prepared


def _linear_rows(shape="s", dur=1.0, tp=0.5, factors=(1.0, 2.0, 3.0, 4.0, 5.0), scale=1.0):
    rows = []
    for factor in factors:
        for node, per_factor in (("N0", 10.0), ("N1", 20.0)):
            rows.append(
                dict(shape_id=shape, node_id=node, factor_mult=factor,
                     vol=per_factor * factor * scale,
                     duracion_horas=dur, tiempo_al_pico_h=tp)
            )
    return rows


def _predictions(prepared, oof):
    keys = prepared.keys.loc[oof["sample_idx"]]
    out = oof.assign(node=keys["node_id"].to_numpy(), factor=keys["factor_mult"].to_numpy())
    return out.set_index(["node", "factor"])["y_pred_reg"]


def test_interpolation_reproduces_a_linear_function_in_the_interior():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    oof = baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0)
    predicted = _predictions(prepared, oof)

    assert predicted[("N0", 3.0)] == pytest.approx(30.0)
    assert predicted[("N1", 3.0)] == pytest.approx(60.0)
    assert predicted[("N0", 2.0)] == pytest.approx(20.0)


def test_interpolation_uses_the_nearest_neighbour_at_the_extremes():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    predicted = _predictions(prepared, baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0))

    assert predicted[("N0", 1.0)] == pytest.approx(20.0)   # vecino f=2.0
    assert predicted[("N0", 5.0)] == pytest.approx(40.0)   # vecino f=4.0


def test_interpolation_label_follows_the_threshold():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    oof = baseline_oof(prepared, INTERPOLACION_FACTOR, 35.0)
    keys = prepared.keys.loc[oof["sample_idx"]]
    labels = oof.assign(node=keys["node_id"].to_numpy(), factor=keys["factor_mult"].to_numpy())
    labels = labels.set_index(["node", "factor"])["y_pred_clf"]

    assert labels[("N0", 3.0)] == 0     # 30 < 35
    assert labels[("N1", 3.0)] == 1     # 60 >= 35


def test_baseline_oof_has_the_oof_contract_and_is_scored_by_score_predictions():
    prepared = make_prepared(_linear_rows(), ["LOSO"])

    oof, metrics = evaluate_baseline(prepared, INTERPOLACION_FACTOR, 1.0)

    assert list(oof.columns) == list(OOF_COLUMNS)
    assert set(metrics) == {"LOSO"}
    assert metrics["LOSO"]["classifier"]["f1"] == pytest.approx(1.0)


def test_interpolation_does_not_apply_to_shape_protocols_and_is_omitted():
    rows = _linear_rows("A") + _linear_rows("B", scale=2.0)
    prepared = make_prepared(rows, ["LOSO", "LeaveOneShapeOut"])

    oof = baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0)

    assert set(oof["protocol"]) == {"LOSO"}


def test_nearest_shape_only_applies_to_shape_protocols():
    rows = _linear_rows("A") + _linear_rows("B", scale=2.0)
    prepared = make_prepared(rows, ["LOSO", "LeaveOneShapeOut"])

    oof = baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)

    assert set(oof["protocol"]) == {"LeaveOneShapeOut"}


def test_nearest_shape_borrows_the_volume_of_the_closest_training_shape():
    rows = (
        _linear_rows("A", dur=1.0, tp=0.2, factors=(1.0, 2.0), scale=5.0)
        + _linear_rows("B", dur=1.1, tp=0.2, factors=(1.0, 2.0), scale=7.0)
        + _linear_rows("C", dur=10.0, tp=5.0, factors=(1.0, 2.0), scale=100.0)
    )
    prepared = make_prepared(rows, ["LeaveOneShapeOut"])
    oof = baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)

    keys = prepared.keys.loc[oof["sample_idx"]]
    table = oof.assign(
        shape=keys["shape_id"].to_numpy(),
        node=keys["node_id"].to_numpy(),
        factor=keys["factor_mult"].to_numpy(),
    ).set_index(["shape", "node", "factor"])["y_pred_reg"]

    # B se parece a A, A se parece a B, C se parece a B (la más cercana en duración/pico)
    assert table[("B", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 5.0)    # volumen de A
    assert table[("A", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 7.0)    # volumen de B
    assert table[("C", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 7.0)    # volumen de B


def test_nearest_shape_baseline_is_scored_with_the_shape_breakdown():
    rows = (
        _linear_rows("A", dur=1.0, tp=0.2, factors=(1.0, 2.0), scale=5.0)
        + _linear_rows("B", dur=1.1, tp=0.2, factors=(1.0, 2.0), scale=7.0)
    )
    prepared = make_prepared(rows, ["LeaveOneShapeOut"])

    _, metrics = evaluate_baseline(prepared, FORMA_MAS_CERCANA, 1.0)

    assert set(metrics["LeaveOneShapeOut"]["by_shape"]) == {"A", "B"}


def test_a_baseline_with_no_applicable_protocol_raises_the_dedicated_error():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    with pytest.raises(BaselineNotApplicable, match="LOSO"):
        baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)


def test_unknown_baseline_name_is_rejected():
    prepared = make_prepared(_linear_rows(), ["LOSO"])
    with pytest.raises(ValueError, match="desconocida"):
        baseline_oof(prepared, "baseline_inventada", 1.0)


def test_every_protocol_has_an_interpretation():
    for protocol in ("LOSO", "GroupKFold5", "LeaveOneShapeOut"):
        assert PROTOCOL_INTERPRETATION[protocol]
    assert set(BASELINES) == {INTERPOLACION_FACTOR, FORMA_MAS_CERCANA}


_SHAPE_COLUMNS = ["duracion_horas", "tiempo_al_pico_h"]


def _without_shape_columns(prepared):
    X = prepared.X.drop(columns=_SHAPE_COLUMNS)
    return dataclasses.replace(
        prepared, X=X, manifest={**prepared.manifest, "feature_columns": list(X.columns)}
    )


def _three_tied_shapes():
    rows = (
        _linear_rows("A", factors=(1.0, 2.0), scale=5.0)
        + _linear_rows("B", factors=(1.0, 2.0), scale=7.0)
        + _linear_rows("C", factors=(1.0, 2.0), scale=100.0)
    )
    return make_prepared(rows, ["LeaveOneShapeOut"])


def _borrowed(prepared, oof):
    keys = prepared.keys.loc[oof["sample_idx"]]
    table = oof.assign(
        shape=keys["shape_id"].to_numpy(),
        node=keys["node_id"].to_numpy(),
        factor=keys["factor_mult"].to_numpy(),
    )
    return table.set_index(["shape", "node", "factor"])["y_pred_reg"]


def test_explicit_shape_descriptors_override_the_features_in_x():
    prepared = _three_tied_shapes()   # X: las tres formas con descriptores idénticos
    descriptors = pd.DataFrame(
        {"d1": [0.0, 10.0, 11.0], "d2": [0.0, 0.0, 0.0]}, index=["A", "B", "C"]
    )
    # A: la más cercana es B; B: la más cercana es C; C: la más cercana es B
    oof = baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0, shape_descriptors=descriptors)
    table = _borrowed(prepared, oof)

    assert table[("A", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 7.0)     # volumen de B
    assert table[("B", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 100.0)   # volumen de C
    assert table[("C", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 7.0)     # volumen de B


def test_explicit_descriptors_can_make_the_closest_shape_differ_from_the_tie_break():
    prepared = _three_tied_shapes()
    # Con X idéntico el desempate daría siempre la primera forma de entrenamiento.
    descriptors = pd.DataFrame(
        {"d1": [0.0, 50.0, 1.0]}, index=["A", "B", "C"]
    )
    table = _borrowed(prepared, baseline_oof(
        prepared, FORMA_MAS_CERCANA, 1.0, shape_descriptors=descriptors))

    assert table[("C", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 5.0)     # C -> A
    assert table[("A", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 100.0)   # A -> C
    assert table[("B", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 100.0)   # B -> C


def test_nearest_shape_without_shape_columns_in_x_is_not_applicable():
    prepared = _without_shape_columns(_three_tied_shapes())

    with pytest.raises(BaselineNotApplicable, match="shape_descriptors"):
        baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)


def test_nearest_shape_without_shape_columns_in_x_works_with_explicit_descriptors():
    prepared = _without_shape_columns(_three_tied_shapes())
    descriptors = pd.DataFrame({"d1": [0.0, 50.0, 1.0]}, index=["A", "B", "C"])

    oof, metrics = evaluate_baseline(
        prepared, FORMA_MAS_CERCANA, 1.0, shape_descriptors=descriptors
    )

    assert set(oof["protocol"]) == {"LeaveOneShapeOut"}
    assert set(metrics["LeaveOneShapeOut"]["by_shape"]) == {"A", "B", "C"}


def test_descriptor_table_missing_a_shape_is_rejected():
    prepared = _three_tied_shapes()
    descriptors = pd.DataFrame({"d1": [0.0, 1.0]}, index=["A", "B"])

    with pytest.raises(ValueError, match="C"):
        baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0, shape_descriptors=descriptors)


def test_interpolation_does_not_need_shape_columns_in_x():
    prepared = _without_shape_columns(make_prepared(_linear_rows(), ["LOSO"]))

    predicted = _predictions(prepared, baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0))

    assert predicted[("N0", 3.0)] == pytest.approx(30.0)


def test_descriptor_table_with_nan_is_rejected_naming_shape_and_column():
    prepared = _three_tied_shapes()
    descriptors = pd.DataFrame(
        {"d1": [0.0, np.nan, 2.0], "d2": [0.0, 1.0, 2.0]}, index=["A", "B", "C"]
    )

    with pytest.raises(ValueError, match=r"NaN.*'B'.*'d1'|NaN.*B.*d1"):
        baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0, shape_descriptors=descriptors)


def test_descriptor_table_with_duplicate_shapes_is_rejected_naming_them():
    prepared = _three_tied_shapes()
    descriptors = pd.DataFrame(
        {"d1": [0.0, 1.0, 2.0, 3.0]}, index=["A", "B", "C", "C"]
    )

    with pytest.raises(ValueError, match=r"duplicad.*'C'|duplicad.*\bC\b"):
        baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0, shape_descriptors=descriptors)


def test_descriptor_table_without_columns_is_rejected():
    prepared = _three_tied_shapes()
    descriptors = pd.DataFrame(index=["A", "B", "C"])

    with pytest.raises(ValueError, match="sin columnas"):
        baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0, shape_descriptors=descriptors)


def test_nan_in_the_shape_features_of_x_is_rejected_naming_shape_and_column():
    prepared = _three_tied_shapes()
    X = prepared.X.copy()
    rows_b = prepared.keys.index[prepared.keys["shape_id"] == "B"]
    X.loc[rows_b[0], "tiempo_al_pico_h"] = np.nan   # solo una fila de la forma B
    prepared = dataclasses.replace(prepared, X=X)

    with pytest.raises(ValueError, match=r"NaN.*B.*tiempo_al_pico_h"):
        baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)


# --- mallas incompletas (celdas retiradas por el filtro de continuidad) -------------

def _grid_rows(volume_of, factors=(1.0, 2.0, 3.0, 4.0, 5.0), nodes=("N0", "N1"),
               shape="s", omit=()):
    rows = []
    for factor in factors:
        for node in nodes:
            if (node, factor) in omit:
                continue
            rows.append(
                dict(shape_id=shape, node_id=node, factor_mult=factor,
                     vol=volume_of(node, factor),
                     duracion_horas=1.0, tiempo_al_pico_h=0.5)
            )
    return rows


def _volume(node, factor):
    return 10.0 * factor if node == "N0" else factor ** 2


def test_interpolation_uses_the_series_own_neighbours_when_a_middle_factor_is_missing():
    # N0 no tiene la celda f=3.0; N1 sí. Retenido f=2.0: el entrenamiento a nivel de
    # fold tiene {1,3,4,5}, pero N0 interpola entre 1.0 y 4.0 (pesos 2/3 y 1/3).
    prepared = make_prepared(_grid_rows(_volume, omit={("N0", 3.0)}), ["LOSO"])

    predicted = _predictions(prepared, baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0))

    assert predicted[("N0", 2.0)] == pytest.approx(10.0 + (1.0 / 3.0) * (40.0 - 10.0))  # 1 y 4 -> 20
    assert predicted[("N1", 2.0)] == pytest.approx(1.0 + 0.5 * (9.0 - 1.0))              # 1 y 3 -> 5
    # N1 con la malla completa no cambia; N0 con f=4.0 retenido usa 2.0 y 5.0
    assert predicted[("N0", 4.0)] == pytest.approx(20.0 + (2.0 / 3.0) * (50.0 - 20.0))   # 40


def test_interpolation_falls_back_to_one_side_when_an_end_factor_is_missing():
    prepared = make_prepared(_grid_rows(_volume, omit={("N0", 5.0)}), ["LOSO"])

    predicted = _predictions(prepared, baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0))

    # f=4.0 retenido: N0 no tiene 5.0 en el train -> sólo el vecino inferior (3.0)
    assert predicted[("N0", 4.0)] == pytest.approx(30.0)
    # N1 conserva la interpolación normal entre 3.0 y 5.0
    assert predicted[("N1", 4.0)] == pytest.approx(9.0 + 0.5 * (25.0 - 9.0))


def test_interpolation_raises_for_a_series_without_any_training_value():
    omit = {("N0", f) for f in (1.0, 2.0, 4.0, 5.0)}   # N0 sólo existe en f=3.0
    prepared = make_prepared(_grid_rows(_volume, omit=omit), ["LOSO"])

    with pytest.raises(ValueError, match=r"s\|N0"):
        baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0)


def test_interpolation_reports_duplicate_cells_with_a_domain_message():
    rows = _grid_rows(_volume) + _grid_rows(_volume, factors=(2.0,), nodes=("N0",))
    prepared = make_prepared(rows, ["LOSO"])

    with pytest.raises(ValueError, match=r"duplicad.*s\|N0"):
        baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0)


def test_interpolation_reports_nan_volumes_in_train():
    rows = _grid_rows(_volume)
    rows[0]["vol"] = float("nan")
    prepared = make_prepared(rows, ["LOSO"])

    with pytest.raises(ValueError, match="NaN"):
        baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0)


def test_fallback_counters_for_the_interpolation_baseline():
    prepared = make_prepared(_grid_rows(_volume, omit={("N0", 3.0)}), ["LOSO"])

    oof = baseline_oof(prepared, INTERPOLACION_FACTOR, 1.0)

    # 9 filas de test (10 celdas menos la ausente). N0 con f=2.0 y f=4.0 usan un
    # vecino que no es el del entrenamiento a nivel de fold; el resto no.
    assert oof.attrs["n_rows"] == 9
    assert oof.attrs["fallback_rows"] == 2


def _three_shapes_without_a_cell():
    rows = (
        _linear_rows("A", dur=1.0, tp=0.2, factors=(1.0, 2.0), scale=5.0)
        + _linear_rows("B", dur=1.1, tp=0.2, factors=(1.0, 2.0), scale=7.0)
        + _linear_rows("C", dur=10.0, tp=5.0, factors=(1.0, 2.0), scale=100.0)
    )
    rows = [
        r for r in rows
        if not (r["shape_id"] == "B" and r["node_id"] == "N0" and r["factor_mult"] == 2.0)
    ]
    return make_prepared(rows, ["LeaveOneShapeOut"])


def test_nearest_shape_walks_down_the_ranking_when_the_nearest_lacks_the_cell():
    prepared = _three_shapes_without_a_cell()

    table = _borrowed(prepared, baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0))

    # A: la más cercana es B, que no tiene (N0, 2.0) -> segunda más cercana, C
    assert table[("A", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 100.0)
    assert table[("A", "N1", 2.0)] == pytest.approx(20.0 * 2.0 * 7.0)     # B sí la tiene
    # C: la más cercana es B (sin la celda) -> A
    assert table[("C", "N0", 2.0)] == pytest.approx(10.0 * 2.0 * 5.0)
    assert table[("C", "N1", 2.0)] == pytest.approx(20.0 * 2.0 * 7.0)     # B
    # B conserva A como más cercana
    assert table[("B", "N1", 2.0)] == pytest.approx(20.0 * 2.0 * 5.0)


def test_fallback_counters_for_the_nearest_shape_baseline():
    prepared = _three_shapes_without_a_cell()

    oof = baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)

    assert oof.attrs["n_rows"] == 11
    assert oof.attrs["fallback_rows"] == 2


def test_nearest_shape_raises_when_no_training_shape_has_the_cell():
    rows = _linear_rows("A", factors=(1.0, 2.0)) + [
        r for r in _linear_rows("B", factors=(1.0, 2.0))
        if not (r["node_id"] == "N0" and r["factor_mult"] == 2.0)
    ]
    prepared = make_prepared(rows, ["LeaveOneShapeOut"])

    with pytest.raises(ValueError, match=r"ninguna forma.*N0"):
        baseline_oof(prepared, FORMA_MAS_CERCANA, 1.0)
