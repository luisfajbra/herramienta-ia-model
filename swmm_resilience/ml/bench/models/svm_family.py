"""Familia SVM: SVC para clasificar, SVR para regresar.

probability=True es obligatorio: el evaluador llama predict_proba para
calcular AUC-ROC. Encarece el ajuste (SVC hace Platt scaling interno) pero
sin ello esta familia no sería comparable con el resto.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from ....config import ML_RANDOM_STATE
from ..param_filter import filter_estimator_params

FAMILY = "svm"
SCALE_FEATURES = True

_CLASSIFIER_DEFAULTS = {"C": 10.0, "kernel": "rbf", "gamma": "scale"}
_REGRESSOR_DEFAULTS = {"C": 10.0, "kernel": "rbf", "epsilon": 0.1}


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    settings = {
        **_CLASSIFIER_DEFAULTS,
        **filter_estimator_params(params, exclude={"epsilon"}),
    }
    model = SVC(
        **settings,
        probability=True,
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
        random_state=ML_RANDOM_STATE,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_regressor(params: dict) -> Pipeline:
    settings = {**_REGRESSOR_DEFAULTS, **filter_estimator_params(params)}
    model = SVR(**settings)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": "standard", "pca": None}
