"""SVM aproximado: Nystroem RBF + SVM lineal (hinge) + calibracion de Platt."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from swmm_resilience.ml.bench.registry import get_family

PARAMS = {"gamma": 0.5, "n_components": 40, "alpha": 1e-4, "max_iter": 500, "tol": 1e-3}
svm = get_family("svm")


def _frame(X):
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])


def _separable(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return _frame(X), y


def _circle(n=400, seed=1):
    """Clase 1 dentro de un circulo: no separable linealmente."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = ((X**2).sum(axis=1) < 1.5).astype(int)
    return _frame(X), y


def test_classifier_is_nystroem_then_calibrated_hinge_svm():
    pipeline = svm.build_classifier(PARAMS, scale_pos_weight=4.0)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.steps[0][1], SimpleImputer)
    assert any(isinstance(s, StandardScaler) for _, s in pipeline.steps)
    kernel = pipeline.named_steps["kernel_approx"]
    assert isinstance(kernel, Nystroem)
    assert kernel.kernel == "rbf"
    assert kernel.gamma == pytest.approx(0.5)
    assert kernel.n_components == 40

    last = pipeline.steps[-1][1]
    assert isinstance(last, CalibratedClassifierCV)
    assert last.method == "sigmoid"
    assert last.cv == 3
    base = last.estimator
    assert isinstance(base, SGDClassifier)
    assert base.loss == "hinge"
    assert base.alpha == pytest.approx(1e-4)
    assert base.max_iter == 500
    assert base.class_weight == {0: 1.0, 1: pytest.approx(4.0)}


def test_classifier_defaults_are_preserved():
    pipeline = svm.build_classifier({}, scale_pos_weight=1.0)
    kernel = pipeline.named_steps["kernel_approx"]
    base = pipeline.steps[-1][1].estimator
    assert kernel.gamma == pytest.approx(0.05)
    assert kernel.n_components == 200
    assert base.alpha == pytest.approx(1e-4)
    assert base.max_iter == 2000
    assert base.tol == pytest.approx(1e-3)


def test_regressor_is_unchanged_linear_epsilon_insensitive_svr():
    pipeline = svm.build_regressor({**PARAMS, "epsilon": 0.1})
    assert isinstance(pipeline.named_steps["kernel_approx"], Nystroem)
    model = pipeline.steps[-1][1]
    assert isinstance(model, SGDRegressor)
    assert model.loss == "epsilon_insensitive"
    assert model.epsilon == pytest.approx(0.1)


def test_predict_proba_has_two_columns_summing_to_one_within_unit_interval():
    X, y = _separable()
    pipeline = svm.build_classifier(PARAMS, scale_pos_weight=1.0).fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_predict_is_consistent_with_calibrated_probabilities():
    """y_pred == (proba >= 0.5) salvo empates.

    Tolerancia: se ignoran las filas con |proba - 0.5| < 1e-6 (empates
    numericos de la frontera); en el resto la coincidencia debe ser exacta.
    """
    X, y = _separable()
    pipeline = svm.build_classifier(PARAMS, scale_pos_weight=1.0).fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    no_tie = np.abs(proba - 0.5) > 1e-6
    assert no_tie.sum() > 0.9 * len(X)
    assert (pred[no_tie] == (proba[no_tie] >= 0.5).astype(int)).all()


def test_classifier_handles_nan_feature_column_and_imbalanced_classes():
    X, y = _separable(n=400)
    rng = np.random.default_rng(3)
    # desbalance ~10 % de positivos
    y = (rng.uniform(size=len(y)) < 0.1).astype(int) | (
        (X["f0"] > 1.6).to_numpy().astype(int)
    )
    X = X.copy()
    X.loc[::7, "f2"] = np.nan
    pipeline = svm.build_classifier(PARAMS, scale_pos_weight=8.0).fit(X, y)
    proba = pipeline.predict_proba(X)
    assert np.isfinite(proba).all()
    assert set(np.unique(pipeline.predict(X))) <= {0, 1}


def test_same_seed_gives_identical_probabilities():
    X, y = _separable()
    first = svm.build_classifier(PARAMS, scale_pos_weight=2.0).fit(X, y)
    second = svm.build_classifier(PARAMS, scale_pos_weight=2.0).fit(X, y)
    assert np.array_equal(first.predict_proba(X), second.predict_proba(X))


def test_unknown_hyperparameter_raises():
    with pytest.raises(TypeError):
        svm.build_classifier({**PARAMS, "no_existe": 1}, scale_pos_weight=1.0)


def test_epsilon_is_regressor_only_and_never_reaches_the_classifier():
    """Como hasta ahora, el clasificador descarta ``epsilon`` (no falla): no
    debe llegar al SGDClassifier, que conserva su valor por defecto."""
    pipeline = svm.build_classifier({**PARAMS, "epsilon": 0.9}, scale_pos_weight=1.0)
    assert pipeline.steps[-1][1].estimator.epsilon == pytest.approx(0.1)
    assert svm.build_regressor({**PARAMS, "epsilon": 0.9}).steps[-1][1].epsilon == (
        pytest.approx(0.9)
    )


def test_separates_a_linearly_separable_problem():
    X, y = _separable(n=400)
    pipeline = svm.build_classifier(PARAMS, scale_pos_weight=1.0).fit(X, y)
    assert roc_auc_score(y, pipeline.predict_proba(X)[:, 1]) > 0.95


def test_kernel_map_beats_a_linear_baseline_on_a_nonlinear_problem():
    from sklearn.linear_model import LogisticRegression

    X, y = _circle()
    svm_auc = roc_auc_score(
        y,
        svm.build_classifier(PARAMS, scale_pos_weight=1.0).fit(X, y).predict_proba(X)[:, 1],
    )
    linear = LogisticRegression().fit(StandardScaler().fit_transform(X), y)
    linear_auc = roc_auc_score(
        y, linear.predict_proba(StandardScaler().fit_transform(X))[:, 1]
    )
    assert svm_auc > 0.95
    assert svm_auc > linear_auc + 0.2


def test_descriptor_declares_approximate_svm_for_the_classifier():
    descriptor = svm.preprocessing_descriptor(PARAMS)
    assert json.dumps(descriptor)
    assert descriptor["model_type"] == "svm_aproximado_nystroem"
    assert descriptor["loss"] == "hinge"
    assert descriptor["probability_calibration"] == "platt_sigmoid_cv3"
    assert descriptor["imputer"] == "median"
    assert descriptor["scaler"] == "standard"
    assert descriptor["pca"] is None
    assert descriptor["kernel_approx"] == {
        "method": "Nystroem",
        "kernel": "rbf",
        "gamma": 0.5,
        "n_components": 40,
    }


def test_descriptor_of_the_regressor_has_no_probability_calibration():
    descriptor = svm.preprocessing_descriptor({**PARAMS, "epsilon": 0.1})
    assert descriptor["model_type"] == "svm_aproximado_nystroem"
    assert descriptor["loss"] == "epsilon_insensitive"
    assert "probability_calibration" not in descriptor


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_platt_calibration_neutralizes_most_of_class_weight_on_probabilities(seed):
    """Sesgo conocido (R17), documentado en el docstring del modulo.

    class_weight moldea el margen, pero la calibracion de Platt sobre folds
    sin ponderar quita casi todo su efecto en la probabilidad: la media de
    predict_proba apenas cambia entre scale_pos_weight=1 y 8 (en 5 semillas
    observadas el cambio maximo fue ~0.004; el margen de 0.05 es holgado).
    """
    rng = np.random.default_rng(seed)
    X = _frame(rng.normal(size=(600, 3)))
    y = ((X["f0"] + 0.8 * rng.normal(size=600)) > 0.5).astype(int).to_numpy()
    means = [
        svm.build_classifier(PARAMS, scale_pos_weight=w).fit(X, y).predict_proba(X)[:, 1].mean()
        for w in (1.0, 8.0)
    ]
    assert abs(means[0] - means[1]) < 0.05


def test_fitted_classifier_and_regressor_survive_a_joblib_round_trip(tmp_path):
    """train.py persiste candidatos con joblib; CalibratedClassifierCV es un
    objeto nuevo dentro del pickle."""
    import joblib

    X, y = _separable(n=200)
    classifier = svm.build_classifier(PARAMS, scale_pos_weight=2.0).fit(X, y)
    regressor = svm.build_regressor({**PARAMS, "epsilon": 0.1}).fit(X, np.log1p(np.abs(X["f0"])))

    joblib.dump(classifier, tmp_path / "clf.joblib")
    joblib.dump(regressor, tmp_path / "reg.joblib")
    classifier_back = joblib.load(tmp_path / "clf.joblib")
    regressor_back = joblib.load(tmp_path / "reg.joblib")

    assert np.array_equal(classifier.predict_proba(X), classifier_back.predict_proba(X))
    assert np.array_equal(classifier.predict(X), classifier_back.predict(X))
    assert np.array_equal(regressor.predict(X), regressor_back.predict(X))


def test_descriptor_limitation_regressor_params_without_epsilon_look_like_classifier():
    """Limitacion documentada: preprocessing_descriptor solo recibe ``params``
    y distingue el regresor por la presencia de ``epsilon``. Un bloque de
    regresor sin ``epsilon`` se describiria como clasificador (hinge + Platt).
    En el flujo real el regresor siempre lleva ``epsilon`` (config.yaml y el
    espacio de Optuna lo incluyen)."""
    descriptor = svm.preprocessing_descriptor(PARAMS)  # sin epsilon
    assert descriptor["loss"] == "hinge"
    assert descriptor["probability_calibration"] == "platt_sigmoid_cv3"
