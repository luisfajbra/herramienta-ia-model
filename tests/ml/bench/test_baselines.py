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
