import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.evaluate import ProvenanceMismatchError, score_predictions
from swmm_resilience.ml.bench.schemas import (
    FOLD_COLUMNS,
    KEY_COLUMNS,
    OOF_COLUMNS,
    PreparedDataset,
)
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17
from tests.ml.bench.prepared_factory import make_prepared, perfect_oof


@pytest.fixture
def prepared() -> PreparedDataset:
    """6 muestras, 2 factores, LOSO -> 2 folds."""
    n = 6
    keys = pd.DataFrame(
        {
            "run_id": [1, 1, 1, 2, 2, 2],
            "network_id": [1] * n,
            "scenario_id": [1, 1, 1, 2, 2, 2],
            "scenario_key": ["base@1.0"] * 3 + ["base@2.0"] * 3,
            "scenario_kind": ["base"] * n,
            "node_id": ["N0", "N1", "N2"] * 2,
            "factor_mult": [1.0] * 3 + [2.0] * 3,
            "shape_id": ["base"] * n,
        },
        columns=list(KEY_COLUMNS),
    )
    X = pd.DataFrame(
        {col: [float(i) for i in range(n)] for col in FEATURE_COLUMNS_V17}
    )
    folds = pd.DataFrame(
        [
            {
                "protocol": "LOSO",
                "fold_id": 0,
                "sample_idx": i,
                "split": "train" if i >= 3 else "test",
                "group_column": "factor_mult",
                "held_out_groups": '["1.0"]',
            }
            for i in range(n)
        ]
        + [
            {
                "protocol": "LOSO",
                "fold_id": 1,
                "sample_idx": i,
                "split": "train" if i < 3 else "test",
                "group_column": "factor_mult",
                "held_out_groups": '["2.0"]',
            }
            for i in range(n)
        ],
        columns=list(FOLD_COLUMNS),
    )
    return PreparedDataset(
        prep_id="0123456789abcdef",
        keys=keys,
        X=X,
        y_clf=pd.Series([0, 1, 1, 0, 1, 1], name="inunda"),
        y_reg=pd.Series([0.0, 10.0, 20.0, 0.0, 30.0, 40.0], name="vol_inundacion_m3"),
        folds=folds,
        manifest={"prep_id": "0123456789abcdef", "protocols": ["LOSO"]},
        quality={},
    )


def _perfect_oof(prepared: PreparedDataset) -> pd.DataFrame:
    tests = prepared.folds[prepared.folds["split"] == "test"]
    return pd.DataFrame(
        {
            "sample_idx": tests["sample_idx"].values,
            "protocol": tests["protocol"].values,
            "fold_id": tests["fold_id"].values,
            "y_pred_clf": prepared.y_clf.iloc[tests["sample_idx"]].values,
            "y_prob_clf": prepared.y_clf.iloc[tests["sample_idx"]].values.astype(float),
            "y_pred_reg": prepared.y_reg.iloc[tests["sample_idx"]].values,
        },
        columns=list(OOF_COLUMNS),
    )


def test_perfect_predictions_score_perfectly(prepared):
    result = score_predictions(
        prepared, _perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )

    assert result["LOSO"]["classifier"]["f1"] == pytest.approx(1.0)
    assert result["LOSO"]["regressor_oracle"]["nse"] == pytest.approx(1.0)
    assert result["LOSO"]["end_to_end"]["rmse_vol_todos_nodos"] == pytest.approx(0.0)


def test_result_has_the_three_levels_and_the_factor_breakdown(prepared):
    result = score_predictions(
        prepared, _perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )
    assert set(result["LOSO"]) >= {
        "classifier", "regressor_oracle", "end_to_end", "by_factor",
        "confusion_matrix", "classifier_per_fold",
        "classifier_legacy_mean_zero_fill", "end_to_end_pooled", "by_shape",
    }
    assert set(result["LOSO"]["by_factor"]) == {"1.00", "2.00"}


def test_provenance_mismatch_is_rejected(prepared):
    """La garantia de 'mismas entradas': no se puede comparar contra otro dataset."""
    with pytest.raises(ProvenanceMismatchError, match="prep_id"):
        score_predictions(
            prepared, _perfect_oof(prepared), {"prep_id": "deadbeefdeadbeef"}
        )


def test_missing_prep_id_in_provenance_is_rejected(prepared):
    with pytest.raises(ProvenanceMismatchError, match="prep_id"):
        score_predictions(prepared, _perfect_oof(prepared), {})


def test_incomplete_fold_coverage_is_rejected(prepared):
    oof = _perfect_oof(prepared).iloc[:-1]
    with pytest.raises(ValueError, match="cobertura"):
        score_predictions(prepared, oof, {"prep_id": prepared.prep_id})


def test_duplicate_predictions_for_a_sample_are_rejected(prepared):
    oof = _perfect_oof(prepared)
    oof = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicad"):
        score_predictions(prepared, oof, {"prep_id": prepared.prep_id})


def test_missing_oof_column_is_rejected(prepared):
    oof = _perfect_oof(prepared).drop(columns=["y_prob_clf"])
    with pytest.raises(ValueError, match="y_prob_clf"):
        score_predictions(prepared, oof, {"prep_id": prepared.prep_id})


def test_a_model_that_predicts_no_flooding_scores_zero_recall(prepared):
    oof = _perfect_oof(prepared)
    oof["y_pred_clf"] = 0
    oof["y_prob_clf"] = 0.0
    oof["y_pred_reg"] = 0.0

    result = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})

    assert result["LOSO"]["classifier"]["recall"] == pytest.approx(0.0)
    assert result["LOSO"]["end_to_end"]["vol_total_pred_m3"] == pytest.approx(0.0)


def test_regressor_oracle_uses_true_labels_not_predicted(prepared):
    """Nivel 2 filtra con la verdad; un clasificador malo no debe afectarlo."""
    oof = _perfect_oof(prepared)
    oof["y_pred_clf"] = 0        # el clasificador falla del todo
    result = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})

    assert result["LOSO"]["regressor_oracle"]["nse"] == pytest.approx(1.0)
    assert result["LOSO"]["classifier"]["recall"] == pytest.approx(0.0)


def test_end_to_end_gate_suppresses_predicted_volume_when_classifier_says_no_flood(prepared):
    """Nivel 3 enruta con la etiqueta PREDICHA: si el clasificador dice 'no

    inunda', el volumen predicho -por grande que sea- se debe descartar. Sin
    esa compuerta, el volumen de un regresor que predice de todos modos se
    filtraria al total end-to-end, que dejaria de ser el numero honesto que
    se supone que es (a diferencia del oraculo del nivel 2).
    """
    oof = _perfect_oof(prepared)
    oof["y_pred_clf"] = 0          # el clasificador dice "no inunda" en todo
    oof["y_prob_clf"] = 0.0
    oof["y_pred_reg"] = 500.0      # pero el regresor predice un volumen grande

    result = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})

    # Con la compuerta: todo se enruta a 0, el volumen grande nunca se usa.
    assert result["LOSO"]["end_to_end"]["vol_total_pred_m3"] == pytest.approx(0.0)
    assert result["LOSO"]["end_to_end"]["rmse_vol_todos_nodos"] == pytest.approx(
        20.88872897341967
    )
    # Sin la compuerta (usando yr_pred directamente) darian 1500.0 y ~483.52:
    # una diferencia de dos ordenes de magnitud que confirma que la
    # compuerta esta activa.


def test_regressor_oracle_pools_across_folds_not_averaged():
    """Nivel 2 se calcula sobre el POOL concatenado, no el promedio por fold.

    Dos folds con conteos de filas inundadas distintos (2 vs 1) y escalas de
    error distintas (~2 vs ~100) hacen que "concatenar y luego calcular" y
    "calcular por fold y promediar" den numeros DEMOSTRABLEMENTE distintos.
    Los valores esperados replican los de
    test_pooled_regressor_metrics_concatenate_before_computing (Task 7,
    tests/ml/bench/test_metrics.py), pero aqui se llega a ellos por el
    camino completo de score_predictions, no llamando al helper de metrics.py
    directamente.
    """
    n = 3
    keys = pd.DataFrame(
        {
            "run_id": [1, 1, 2],
            "network_id": [1] * n,
            "scenario_id": [1, 1, 2],
            "scenario_key": ["base@1.0"] * 2 + ["base@2.0"],
            "scenario_kind": ["base"] * n,
            "node_id": ["N0", "N1", "N0"],
            "factor_mult": [1.0, 1.0, 2.0],
            "shape_id": ["base"] * n,
        },
        columns=list(KEY_COLUMNS),
    )
    X = pd.DataFrame(
        {col: [float(i) for i in range(n)] for col in FEATURE_COLUMNS_V17}
    )
    folds = pd.DataFrame(
        [
            {"protocol": "LOSO", "fold_id": 0, "sample_idx": 0, "split": "test", "group_column": "factor_mult", "held_out_groups": '["1.0"]'},
            {"protocol": "LOSO", "fold_id": 0, "sample_idx": 1, "split": "test", "group_column": "factor_mult", "held_out_groups": '["1.0"]'},
            {"protocol": "LOSO", "fold_id": 0, "sample_idx": 2, "split": "train", "group_column": "factor_mult", "held_out_groups": '["1.0"]'},
            {"protocol": "LOSO", "fold_id": 1, "sample_idx": 0, "split": "train", "group_column": "factor_mult", "held_out_groups": '["2.0"]'},
            {"protocol": "LOSO", "fold_id": 1, "sample_idx": 1, "split": "train", "group_column": "factor_mult", "held_out_groups": '["2.0"]'},
            {"protocol": "LOSO", "fold_id": 1, "sample_idx": 2, "split": "test", "group_column": "factor_mult", "held_out_groups": '["2.0"]'},
        ],
        columns=list(FOLD_COLUMNS),
    )
    prepared = PreparedDataset(
        prep_id="pool_vs_mean",
        keys=keys,
        X=X,
        y_clf=pd.Series([1, 1, 1], name="inunda"),
        y_reg=pd.Series([10.0, 20.0, 1000.0], name="vol_inundacion_m3"),
        folds=folds,
        manifest={"prep_id": "pool_vs_mean", "protocols": ["LOSO"]},
        quality={},
    )
    oof = pd.DataFrame(
        {
            "sample_idx": [0, 1, 2],
            "protocol": ["LOSO"] * 3,
            "fold_id": [0, 0, 1],
            "y_pred_clf": [1, 1, 1],
            "y_prob_clf": [1.0, 1.0, 1.0],
            "y_pred_reg": [12.0, 18.0, 900.0],
        },
        columns=list(OOF_COLUMNS),
    )

    result = score_predictions(prepared, oof, {"prep_id": "pool_vs_mean"})

    # Pooled (correcto): concatenar [10,20,1000] vs [12,18,900] y calcular una vez.
    assert result["LOSO"]["regressor_oracle"]["rmse"] == pytest.approx(57.758116312774604)
    assert result["LOSO"]["regressor_oracle"]["nse"] == pytest.approx(0.9845284963413378)
    # Promediado por fold (mutante) daria rmse=51.0, nse=0.42 -- muy distinto.


def test_by_factor_values_are_correct_not_just_the_key_set():
    """by_factor no es solo el conjunto de llaves: los NUMEROS por factor

    tienen que ser los correctos, calculados con f1 (no otra metrica de
    clasificador), con la compuerta de enrutamiento del nivel 3 (no
    yr_pred crudo), y con mean_metrics (no pooled_regressor_metrics, que
    aqui ni siquiera aplica -- by_factor no tiene nivel de oraculo).

    Un solo fold con dos grupos de factor, para poder calcular las
    esperadas a mano sin necesidad de promediar entre folds:

    Grupo A (factor 1.00), 3 filas:
      fila0: y_true_clf=1, y_pred_clf=1, y_true_reg=10, y_pred_reg=10  (TP)
      fila1: y_true_clf=0, y_pred_clf=0, y_true_reg=0,  y_pred_reg=0   (TN)
      fila2: y_true_clf=1, y_pred_clf=0, y_true_reg=20, y_pred_reg=500 (FN,
             y la compuerta de enrutamiento tiene que descartar ese 500)

      f1: TP=1, FP=0, FN=1 -> precision=1.0, recall=0.5 -> f1=2/3.
      routed = donde(pred_clf==1, y_pred_reg, 0) = [10, 0, 0]  (la fila2
      NO se enruta pese a su y_pred_reg=500, porque pred_clf=0 alli).
      rmse_vol = sqrt(mean(([10,0,20]-[10,0,0])**2)) = sqrt(400/3).

    Grupo B (factor 2.00), 3 filas, predicciones perfectas:
      fila3: y_true_clf=0, y_pred_clf=0, y_true_reg=0,  y_pred_reg=0
      fila4: y_true_clf=1, y_pred_clf=1, y_true_reg=30, y_pred_reg=30
      fila5: y_true_clf=1, y_pred_clf=1, y_true_reg=40, y_pred_reg=40

      f1=1.0 (todo correcto). routed=[0,30,40]=true -> rmse_vol=0.0.

    Los dos grupos tienen f1 y rmse_vol claramente distintos entre si, y
    distintos de 0/1 en el caso de A, para que ningun mutante "de casualidad"
    siga dando el numero correcto.
    """
    n = 6
    keys = pd.DataFrame(
        {
            "run_id": [1, 1, 1, 2, 2, 2],
            "network_id": [1] * n,
            "scenario_id": [1, 1, 1, 2, 2, 2],
            "scenario_key": ["base@1.0"] * 3 + ["base@2.0"] * 3,
            "scenario_kind": ["base"] * n,
            "node_id": ["N0", "N1", "N2"] * 2,
            "factor_mult": [1.0] * 3 + [2.0] * 3,
            "shape_id": ["base"] * n,
        },
        columns=list(KEY_COLUMNS),
    )
    X = pd.DataFrame(
        {col: [float(i) for i in range(n)] for col in FEATURE_COLUMNS_V17}
    )
    folds = pd.DataFrame(
        [
            {
                "protocol": "LOSO",
                "fold_id": fold_id,
                "sample_idx": i,
                "split": (
                    "test"
                    if (fold_id == 0 and i < 3) or (fold_id == 1 and i >= 3)
                    else "train"
                ),
                "group_column": "factor_mult",
                "held_out_groups": f'["{fold_id + 1}.0"]',
            }
            for fold_id in (0, 1)
            for i in range(n)
        ],
        columns=list(FOLD_COLUMNS),
    )
    prepared = PreparedDataset(
        prep_id="by_factor_values",
        keys=keys,
        X=X,
        y_clf=pd.Series([1, 0, 1, 0, 1, 1], name="inunda"),
        y_reg=pd.Series([10.0, 0.0, 20.0, 0.0, 30.0, 40.0], name="vol_inundacion_m3"),
        folds=folds,
        manifest={"prep_id": "by_factor_values", "protocols": ["LOSO"]},
        quality={},
    )
    oof = pd.DataFrame(
        {
            "sample_idx": [0, 1, 2, 3, 4, 5],
            "protocol": ["LOSO"] * n,
            "fold_id": [0, 0, 0, 1, 1, 1],
            "y_pred_clf": [1, 0, 0, 0, 1, 1],
            "y_prob_clf": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "y_pred_reg": [10.0, 0.0, 500.0, 0.0, 30.0, 40.0],
        },
        columns=list(OOF_COLUMNS),
    )

    result = score_predictions(prepared, oof, {"prep_id": "by_factor_values"})
    by_factor = result["LOSO"]["by_factor"]

    assert by_factor["1.00"]["f1"] == pytest.approx(2.0 / 3.0)
    assert by_factor["1.00"]["rmse_vol"] == pytest.approx((400.0 / 3.0) ** 0.5)
    assert by_factor["2.00"]["f1"] == pytest.approx(1.0)
    assert by_factor["2.00"]["rmse_vol"] == pytest.approx(0.0)


def test_each_protocol_is_scored_from_its_own_predictions_not_pooled():
    """El resultado esta indexado por protocolo; LOSO y GroupKFold5 no se

    deben mezclar en un solo balde. Mismas 5 muestras, predicciones
    con etiqueta positiva, predicciones perfectas bajo LOSO (f1=1.0) y
    predicciones "no inunda en todo" bajo GroupKFold5 (f1=0.0). Asi, el
    promedio de F1 por fold conserva el contraste: si el bucle por protocolo
    se pierde y todo se agrupa junto, ninguno de los dos numeros sale limpio
    (y ni siquiera existirian las llaves "LOSO"/"GroupKFold5" por separado).
    """
    n = 5
    keys = pd.DataFrame(
        {
            "run_id": [1] * n,
            "network_id": [1] * n,
            "scenario_id": list(range(1, n + 1)),
            "scenario_key": [f"base@{value}.0" for value in range(1, n + 1)],
            "scenario_kind": ["base"] * n,
            "node_id": [f"N{value}" for value in range(n)],
            "factor_mult": [float(value) for value in range(1, n + 1)],
            "shape_id": ["base"] * n,
        },
        columns=list(KEY_COLUMNS),
    )
    X = pd.DataFrame(
        {col: [float(i) for i in range(n)] for col in FEATURE_COLUMNS_V17}
    )
    folds = pd.DataFrame(
        [
            {
                "protocol": protocol,
                "fold_id": fold_id,
                "sample_idx": i,
                "split": "test" if i == fold_id else "train",
                "group_column": "factor_mult",
                "held_out_groups": f'["{fold_id + 1}.0"]',
            }
            for protocol in ("LOSO", "GroupKFold5")
            for fold_id in range(5)
            for i in range(n)
        ],
        columns=list(FOLD_COLUMNS),
    )
    prepared = PreparedDataset(
        prep_id="two_protocols",
        keys=keys,
        X=X,
        y_clf=pd.Series([1] * n, name="inunda"),
        y_reg=pd.Series([0.0, 10.0, 20.0, 0.0, 30.0], name="vol_inundacion_m3"),
        folds=folds,
        manifest={
            "prep_id": "two_protocols",
            "protocols": ["LOSO", "GroupKFold5"],
        },
        quality={},
    )
    oof = pd.DataFrame(
        {
            "sample_idx": list(range(n)) * 2,
            "protocol": ["LOSO"] * n + ["GroupKFold5"] * n,
            "fold_id": list(range(n)) * 2,
            # LOSO: predicciones perfectas.
            # GroupKFold5: predice "no inunda" en todo -> f1=0.0.
            "y_pred_clf": [1] * n + [0] * n,
            "y_prob_clf": [1.0] * n + [0.0] * n,
            "y_pred_reg": [0.0, 10.0, 20.0, 0.0, 30.0] + [0.0] * n,
        },
        columns=list(OOF_COLUMNS),
    )

    result = score_predictions(prepared, oof, {"prep_id": "two_protocols"})

    assert set(result) == {"LOSO", "GroupKFold5"}
    assert result["LOSO"]["classifier"]["f1"] == pytest.approx(1.0)
    assert result["GroupKFold5"]["classifier"]["f1"] == pytest.approx(0.0)


def _loso_with_an_empty_fold():
    """3 factores en LOSO; el factor 1.0 no tiene ninguna fila inundada."""
    rows = []
    volumes = {
        1.0: [0.0, 0.0, 0.0],
        2.0: [0.0, 10.0, 20.0],
        3.0: [30.0, 40.0, 50.0],
    }
    for factor, vols in volumes.items():
        for node, vol in zip(["N0", "N1", "N2"], vols):
            rows.append(
                dict(shape_id="base", node_id=node, factor_mult=factor, vol=vol,
                     duracion_horas=3.0, tiempo_al_pico_h=0.7)
            )
    return make_prepared(rows, ["LOSO"])


def test_perfect_predictions_score_f1_one_even_with_a_fold_without_flooding():
    """M-03: el valor agregado no se rebaja por un fold sin positivos."""
    prepared = _loso_with_an_empty_fold()
    result = score_predictions(
        prepared, perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )["LOSO"]

    assert result["classifier"]["f1"] == pytest.approx(1.0)
    assert result["classifier"]["specificity"] == pytest.approx(1.0)


def test_the_legacy_zero_fill_mean_is_kept_under_an_explicit_name():
    prepared = _loso_with_an_empty_fold()
    result = score_predictions(
        prepared, perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )["LOSO"]

    # folds: factor 1.0 (sin positivos) -> 0 ; factores 2.0 y 3.0 -> 1.0
    assert result["classifier_legacy_mean_zero_fill"]["f1"] == pytest.approx(2 / 3)
    assert set(result["classifier_legacy_mean_zero_fill"]) == {
        "precision", "recall", "f1", "auc_roc"
    }


def test_per_fold_dispersion_reports_the_undefined_fold():
    prepared = _loso_with_an_empty_fold()
    result = score_predictions(
        prepared, perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )["LOSO"]

    assert result["classifier_per_fold"]["f1"]["n_undefined"] == 1
    assert result["classifier_per_fold"]["f1"]["min"] == pytest.approx(1.0)


def test_confusion_matrix_counts_every_test_row():
    prepared = _loso_with_an_empty_fold()
    result = score_predictions(
        prepared, perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )["LOSO"]

    assert result["confusion_matrix"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 5}


def test_a_false_negative_and_a_false_positive_show_up_in_the_confusion_matrix():
    prepared = _loso_with_an_empty_fold()
    oof = perfect_oof(prepared)
    oof.loc[oof["sample_idx"] == 4, "y_pred_clf"] = 0   # falso negativo (N1, factor 2.0)
    oof.loc[oof["sample_idx"] == 0, "y_pred_clf"] = 1   # falso positivo (N0, factor 1.0)

    result = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})["LOSO"]

    assert result["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 4}
    assert result["classifier"]["recall"] == pytest.approx(4 / 5)
    assert result["classifier"]["precision"] == pytest.approx(4 / 5)


def test_protocols_restricts_the_coverage_required_from_the_oof_frame():
    """Dataset con dos protocolos y un OOF que cubre solo LOSO.

    Sin ``protocols`` faltan los folds de LeaveOneShapeOut y se rechaza; con
    ``protocols=("LOSO",)`` solo se exige y se puntua LOSO.
    """
    rows = []
    for shape in ("A", "B"):
        for factor in (1.0, 2.0):
            for node, vol in zip(["N0", "N1"], [0.0, 10.0 * factor]):
                rows.append(
                    dict(shape_id=shape, node_id=node, factor_mult=factor, vol=vol,
                         duracion_horas=3.0, tiempo_al_pico_h=0.7)
                )
    prepared = make_prepared(rows, ["LOSO", "LeaveOneShapeOut"])
    oof = perfect_oof(prepared)
    oof = oof[oof["protocol"] == "LOSO"].reset_index(drop=True)
    assert set(oof["protocol"]) == {"LOSO"}
    assert set(prepared.folds["protocol"]) == {"LOSO", "LeaveOneShapeOut"}

    with pytest.raises(ValueError, match="faltan"):
        score_predictions(prepared, oof, {"prep_id": prepared.prep_id})

    restricted = score_predictions(
        prepared, oof, {"prep_id": prepared.prep_id}, protocols=("LOSO",)
    )
    assert set(restricted) == {"LOSO"}
    assert "classifier" in restricted["LOSO"]


def test_by_shape_and_pooled_end_to_end_use_the_routed_volume():
    rows = []
    spec = {
        "A": {("N0", 1.0): 0.0, ("N1", 1.0): 10.0, ("N0", 2.0): 0.0, ("N1", 2.0): 20.0},
        "B": {("N0", 1.0): 0.0, ("N1", 1.0): 0.0, ("N0", 2.0): 40.0, ("N1", 2.0): 50.0},
    }
    for shape, cells in spec.items():
        for (node, factor), vol in cells.items():
            rows.append(
                dict(shape_id=shape, node_id=node, factor_mult=factor, vol=vol,
                     duracion_horas=3.0, tiempo_al_pico_h=0.7)
            )
    prepared = make_prepared(rows, ["LeaveOneShapeOut"])
    oof = perfect_oof(prepared)
    shape_b = prepared.keys.loc[oof["sample_idx"], "shape_id"].to_numpy() == "B"
    oof.loc[shape_b, "y_pred_reg"] = oof.loc[shape_b, "y_pred_reg"] * 2.0  # B se sobreestima 2x

    result = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})["LeaveOneShapeOut"]

    assert result["by_shape"]["A"]["error_vol_pct"] == pytest.approx(0.0)
    assert result["by_shape"]["B"]["error_vol_pct"] == pytest.approx(100.0)
    assert result["by_shape"]["B"]["rmse_vol"] == pytest.approx(np.sqrt((40**2 + 50**2) / 4))
    assert result["end_to_end_pooled"]["vol_total_real_m3"] == pytest.approx(120.0)
    assert result["end_to_end_pooled"]["vol_total_pred_m3"] == pytest.approx(210.0)
    assert result["end_to_end_pooled"]["vol_total_error_pct"] == pytest.approx(75.0)
    assert result["end_to_end_pooled"]["rmse_vol_todos_nodos"] == pytest.approx(
        np.sqrt((40**2 + 50**2) / 8)
    )
