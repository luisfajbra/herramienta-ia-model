import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from swmm_resilience.ml.bench.registry import available_families, get_family
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

SCALED_FAMILIES = ("linear", "svm")
UNSCALED_FAMILIES = ("xgboost", "random_forest")
ALL_FAMILIES = SCALED_FAMILIES + UNSCALED_FAMILIES

TINY_PARAMS = {
    "xgboost": {"n_estimators": 5, "max_depth": 2},
    "random_forest": {"n_estimators": 5, "max_depth": 2},
    "linear": {"alpha": 1.0},
    "svm": {"C": 1.0},
}


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS_V17))), columns=list(FEATURE_COLUMNS_V17)
    )
    y_clf = (X["q_pico_nodo"] > 0).astype(int).to_numpy()
    y_reg = np.abs(X["q_pico_nodo"].to_numpy()) * 10.0
    return X, y_clf, y_reg


def test_all_four_families_are_registered():
    assert set(available_families()) == set(ALL_FAMILIES)


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_every_family_exposes_the_same_interface(name):
    family = get_family(name)
    assert family.FAMILY == name
    assert isinstance(family.SCALE_FEATURES, bool)
    assert callable(family.build_classifier)
    assert callable(family.build_regressor)
    assert callable(family.preprocessing_descriptor)


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_every_pipeline_starts_with_a_median_imputer(name):
    pipeline = get_family(name).build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    first = pipeline.steps[0][1]
    assert isinstance(first, SimpleImputer)
    assert first.strategy == "median"


@pytest.mark.parametrize("name", SCALED_FAMILIES)
def test_scaled_families_include_a_standard_scaler(name):
    family = get_family(name)
    assert family.SCALE_FEATURES is True
    pipeline = family.build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    assert any(isinstance(step, StandardScaler) for _, step in pipeline.steps)


@pytest.mark.parametrize("name", UNSCALED_FAMILIES)
def test_tree_families_have_no_scaler(name):
    family = get_family(name)
    assert family.SCALE_FEATURES is False
    pipeline = family.build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    assert not any(isinstance(step, StandardScaler) for _, step in pipeline.steps)


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_no_family_uses_pca(name):
    """PCA era del stack A: degradaba la interpretabilidad y no vuelve."""
    pipeline = get_family(name).build_regressor(TINY_PARAMS[name])
    assert "pca" not in dict(pipeline.named_steps)


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_classifier_fits_and_predicts_binary_labels(name, toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family(name).build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    assert set(np.unique(predictions)) <= {0, 1}
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_regressor_fits_and_predicts_floats(name, toy_data):
    X, _, y_reg = toy_data
    pipeline = get_family(name).build_regressor(TINY_PARAMS[name])
    pipeline.fit(X, np.log1p(y_reg))

    predictions = pipeline.predict(X)
    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_preprocessing_descriptor_is_json_serialisable(name):
    descriptor = get_family(name).preprocessing_descriptor(TINY_PARAMS[name])
    assert json.dumps(descriptor)
    assert set(descriptor) == {"imputer", "scaler", "pca"}


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_families_handle_missing_values_via_the_imputer(name, toy_data):
    X, y_clf, _ = toy_data
    X = X.copy()
    X.loc[0, "diam_max_in"] = np.nan

    pipeline = get_family(name).build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)
    assert len(pipeline.predict(X)) == len(X)


def test_svm_classifier_supports_probability_output():
    """SVC necesita probability=True o predict_proba revienta en el evaluador."""
    pipeline = get_family("svm").build_classifier({"C": 1.0}, scale_pos_weight=1.0)
    assert pipeline.named_steps["model"].probability is True


def test_class_weight_is_used_where_scale_pos_weight_does_not_exist():
    """linear y svm no tienen scale_pos_weight: el desbalance va por class_weight."""
    for name in SCALED_FAMILIES:
        pipeline = get_family(name).build_classifier(TINY_PARAMS[name], scale_pos_weight=4.0)
        weights = pipeline.named_steps["model"].class_weight
        assert weights == {0: 1.0, 1: pytest.approx(4.0)}
