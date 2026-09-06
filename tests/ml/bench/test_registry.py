import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from swmm_resilience.ml.bench.registry import available_families, get_family


def test_xgboost_family_is_registered():
    assert "xgboost" in available_families()


def test_unknown_family_lists_the_available_ones():
    with pytest.raises(ValueError, match="Familia desconocida"):
        get_family("perceptron_de_1958")


def test_xgboost_classifier_is_imputer_plus_model_without_scaler():
    family = get_family("xgboost")
    pipeline = family.build_classifier(
        {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0},
        scale_pos_weight=2.0,
    )

    assert isinstance(pipeline, Pipeline)
    steps = dict(pipeline.named_steps)
    assert isinstance(steps["imputer"], SimpleImputer)
    assert steps["imputer"].strategy == "median"
    assert isinstance(steps["model"], XGBClassifier)
    assert not any(isinstance(step, StandardScaler) for step in steps.values()), (
        "XGBoost debe ir sin escalador: es lo que permite la paridad con trainer.py"
    )


def test_xgboost_regressor_is_imputer_plus_model():
    family = get_family("xgboost")
    pipeline = family.build_regressor(
        {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0}
    )
    assert isinstance(pipeline.named_steps["model"], XGBRegressor)


def test_scale_pos_weight_reaches_the_classifier():
    family = get_family("xgboost")
    pipeline = family.build_classifier({"n_estimators": 5}, scale_pos_weight=3.5)
    assert pipeline.named_steps["model"].scale_pos_weight == pytest.approx(3.5)


def test_family_declares_it_does_not_scale():
    assert get_family("xgboost").SCALE_FEATURES is False


def test_preprocessing_descriptor_is_json_serialisable_and_names_the_steps():
    import json

    descriptor = get_family("xgboost").preprocessing_descriptor({"n_estimators": 5})
    assert json.dumps(descriptor)
    assert descriptor["imputer"] == "median"
    assert descriptor["scaler"] is None


def test_random_state_is_fixed_for_reproducibility():
    from swmm_resilience.config import ML_RANDOM_STATE

    pipeline = get_family("xgboost").build_regressor({"n_estimators": 5})
    assert pipeline.named_steps["model"].random_state == ML_RANDOM_STATE
