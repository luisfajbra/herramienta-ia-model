"""Familia XGBoost.

Va SIN escalador a propósito: los árboles no lo necesitan, y es la condición
para que este candidato reproduzca exactamente las métricas de ml/trainer.py
en el test de paridad.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from ....config import ML_RANDOM_STATE

FAMILY = "xgboost"
SCALE_FEATURES = False

_CLASSIFIER_DEFAULTS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
}
_REGRESSOR_DEFAULTS = dict(_CLASSIFIER_DEFAULTS)


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    settings = {**_CLASSIFIER_DEFAULTS, **params}
    settings.pop("scale_pos_weight", None)
    model = XGBClassifier(
        **settings,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=ML_RANDOM_STATE,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def build_regressor(params: dict) -> Pipeline:
    settings = {**_REGRESSOR_DEFAULTS, **params}
    model = XGBRegressor(**settings, random_state=ML_RANDOM_STATE)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": None, "pca": None}
