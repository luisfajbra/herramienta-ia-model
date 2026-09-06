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
            {"protocol": "LOSO", "fold_id": 0, "sample_idx": i, "split": "train" if i >= 3 else "test"}
            for i in range(n)
        ]
        + [
            {"protocol": "LOSO", "fold_id": 1, "sample_idx": i, "split": "train" if i < 3 else "test"}
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
        manifest={"prep_id": "0123456789abcdef"},
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
    assert set(result["LOSO"]) == {
        "classifier", "regressor_oracle", "end_to_end", "by_factor"
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
