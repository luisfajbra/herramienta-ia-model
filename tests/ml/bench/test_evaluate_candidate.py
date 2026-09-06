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


def test_scale_pos_weight_is_computed_per_fold_not_globally(prepared, monkeypatch):
    """La fuga de balance de clases entre folds debe ser imposible de introducir sin fallar."""
    from swmm_resilience.ml.bench.models import xgboost_family

    recorded: list[float] = []
    original_build_classifier = xgboost_family.build_classifier

    def spy_build_classifier(params, scale_pos_weight):
        recorded.append(scale_pos_weight)
        return original_build_classifier(params, scale_pos_weight)

    monkeypatch.setattr(xgboost_family, "build_classifier", spy_build_classifier)
    evaluate_candidate(prepared, "xgboost", TINY, TINY)

    assert len(recorded) == 3
    assert len(set(recorded)) > 1, (
        "scale_pos_weight salio igual en los tres folds: eso sugiere que se "
        "calculo sobre el dataset completo en vez de sobre el train de cada "
        "fold, lo que filtraria el balance de clases del fold de validacion "
        "hacia el entrenamiento"
    )

    # Valores esperados calculados a mano, sin llamar a evaluate_candidate,
    # a partir de la composicion conocida del fixture csv_shaped_dataset
    # (tests/conftest.py): 2 shapes x 3 factor_mult x 4 nodos; inundan los
    # nodos 0 y 1 cuando factor_mult >= 2.0. Por factor_mult, sobre ambos
    # shapes combinados (8 filas cada uno):
    #   factor 1.0 -> 0 positivos, 8 negativos
    #   factor 2.0 -> 4 positivos, 4 negativos
    #   factor 3.0 -> 4 positivos, 4 negativos
    # LOSO agrupa por factor_mult, así que el train de cada fold es la union
    # de los otros dos factores:
    #   deja fuera 1.0 -> train = {2.0, 3.0} -> 8 pos, 8 neg -> spw = 1.0
    #   deja fuera 2.0 -> train = {1.0, 3.0} -> 4 pos, 12 neg -> spw = 3.0
    #   deja fuera 3.0 -> train = {1.0, 2.0} -> 4 pos, 12 neg -> spw = 3.0
    expected = [1.0, 3.0, 3.0]
    assert sorted(recorded) == pytest.approx(sorted(expected))


def test_the_inverse_log_transform_is_applied_to_regressor_predictions(prepared, monkeypatch):
    """Sin expm1, un regresor que predice log1p(250) reportaria ~5.525, no 250."""
    from swmm_resilience.ml.bench.models import xgboost_family

    class _ConstantLogSpaceRegressor:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), np.log1p(250.0))

    monkeypatch.setattr(
        xgboost_family, "build_regressor", lambda params: _ConstantLogSpaceRegressor()
    )
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)

    assert oof["y_pred_reg"].to_numpy() == pytest.approx(250.0)


def test_negative_regressor_predictions_are_clipped_to_zero(prepared, monkeypatch):
    """expm1(-3.0) es negativo; sin el clip llegaria a score_predictions como volumen negativo."""
    from swmm_resilience.ml.bench.models import xgboost_family

    class _NegativeLogSpaceRegressor:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), -3.0)

    monkeypatch.setattr(
        xgboost_family, "build_regressor", lambda params: _NegativeLogSpaceRegressor()
    )
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)

    assert (oof["y_pred_reg"].to_numpy() == 0.0).all()
