"""Familia lineal: LogisticRegression para clasificar, Ridge para regresar.

Lleva StandardScaler dentro del Pipeline, así que se ajusta con el train de
cada fold. Es el baseline honesto contra el que se mide XGBoost.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ....config import ML_RANDOM_STATE

FAMILY = "linear"
SCALE_FEATURES = True

_CLASSIFIER_DEFAULTS = {"C": 1.0, "max_iter": 5000}
_REGRESSOR_DEFAULTS = {"alpha": 1.0}


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    settings = {**_CLASSIFIER_DEFAULTS, **{k: v for k, v in params.items() if k != "alpha"}}
    model = LogisticRegression(
        **settings,
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
    settings = {**_REGRESSOR_DEFAULTS, **{k: v for k, v in params.items() if k != "C"}}
    model = Ridge(**settings, random_state=ML_RANDOM_STATE)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": "standard", "pca": None}
