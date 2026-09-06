import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from swmm_resilience.ml.bench.registry import available_families, get_family
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

PARAMS = {"hidden_sizes": (16, 8), "epochs": 20, "learning_rate": 0.01, "batch_size": 16}


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(7)
    n = 80
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS_V17))), columns=list(FEATURE_COLUMNS_V17)
    )
    y_clf = (X["q_pico_nodo"] + X["prof_max"] > 0).astype(int).to_numpy()
    y_reg = np.abs(X["q_pico_nodo"].to_numpy()) * 10.0
    return X, y_clf, y_reg


def test_mlp_is_registered_as_a_family():
    assert "mlp" in available_families()
    assert get_family("mlp").SCALE_FEATURES is True


def test_pipeline_has_imputer_scaler_and_model():
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    steps = dict(pipeline.named_steps)
    assert set(steps) == {"imputer", "scaler", "model"}
    assert isinstance(steps["scaler"], StandardScaler)


def test_classifier_fits_and_returns_binary_predictions(toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)

    predictions = pipeline.predict(X)
    assert set(np.unique(predictions)) <= {0, 1}
    assert len(predictions) == len(X)


def test_predict_proba_returns_two_calibrated_columns(toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)

    probabilities = pipeline.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_regressor_fits_and_predicts_finite_values(toy_data):
    X, _, y_reg = toy_data
    pipeline = get_family("mlp").build_regressor(PARAMS)
    pipeline.fit(X, np.log1p(y_reg))

    predictions = pipeline.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_the_same_seed_gives_the_same_predictions(toy_data):
    """Sin esto el MLP no es comparable: cada corrida daria otro numero."""
    X, y_clf, _ = toy_data
    first = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    second = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    first.fit(X, y_clf)
    second.fit(X, y_clf)

    np.testing.assert_allclose(
        first.predict_proba(X)[:, 1], second.predict_proba(X)[:, 1], rtol=1e-6
    )


def test_the_estimator_is_clonable_by_sklearn():
    """Pipeline y cross-validation dependen de get_params/set_params."""
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    cloned = clone(pipeline)
    assert cloned.named_steps["model"].get_params()["epochs"] == PARAMS["epochs"]


def test_scale_pos_weight_reaches_the_loss(toy_data):
    """El desbalance se aplica como pos_weight de BCEWithLogitsLoss."""
    model = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=5.0).named_steps["model"]
    assert model.get_params()["scale_pos_weight"] == pytest.approx(5.0)


def test_training_runs_on_cpu_without_requiring_cuda(toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)
    assert next(pipeline.named_steps["model"].network_.parameters()).device.type == "cpu"


def test_preprocessing_descriptor_declares_the_scaler():
    descriptor = get_family("mlp").preprocessing_descriptor(PARAMS)
    assert descriptor == {"imputer": "median", "scaler": "standard", "pca": None}
