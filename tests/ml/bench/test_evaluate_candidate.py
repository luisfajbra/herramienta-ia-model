import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.evaluate import evaluate_candidate, score_predictions
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.schemas import OOF_COLUMNS

TINY = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0}


@pytest.fixture
def prepared(sql_training_db, tmp_path):
    return prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_oof_frame_has_the_declared_schema(prepared):
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    assert tuple(oof.columns) == OOF_COLUMNS


def test_oof_covers_every_test_row_exactly_once(prepared):
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    expected = prepared.folds[prepared.folds["split"] == "test"]
    assert len(oof) == len(expected)
    assert not oof.duplicated(subset=["protocol", "fold_id", "sample_idx"]).any()


def test_metrics_match_scoring_the_returned_oof(prepared):
    """La capa de conveniencia no debe calcular metricas por su cuenta."""
    oof, metrics = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    recomputed = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})
    assert metrics == recomputed


def test_predicted_volumes_are_never_negative(prepared):
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    assert (oof["y_pred_reg"] >= 0).all()


def test_probabilities_are_in_the_unit_interval(prepared):
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    assert oof["y_prob_clf"].between(0.0, 1.0).all()


def test_running_it_twice_gives_identical_predictions(prepared):
    first, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    second, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.xfail(reason="familia linear llega en la Task 12", strict=False)
def test_the_scaler_is_fitted_inside_the_fold_not_on_the_whole_dataset(prepared, monkeypatch):
    """El nucleo de la garantia anti-fuga, verificado en ejecucion."""
    from sklearn.preprocessing import StandardScaler

    seen_row_counts = []
    original_fit = StandardScaler.fit

    def spy_fit(self, X, y=None, **kwargs):
        seen_row_counts.append(len(X))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spy_fit)
    evaluate_candidate(prepared, "linear", {"alpha": 1.0}, {"alpha": 1.0})

    total_rows = len(prepared.X)
    assert seen_row_counts, "la familia linear deberia ajustar un StandardScaler"
    assert all(count < total_rows for count in seen_row_counts), (
        f"un scaler se ajusto con {max(seen_row_counts)} filas de {total_rows}: "
        "eso es el dataset completo, hay fuga de datos"
    )


def test_a_fold_without_flooded_training_rows_still_produces_predictions(prepared):
    """No debe reventar: predice volumen cero donde no pudo entrenar el regresor."""
    oof, metrics = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    assert len(oof) > 0
    assert "LOSO" in metrics
