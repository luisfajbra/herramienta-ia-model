"""Familia RandomForest. Sin escalador: los árboles no lo necesitan."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ....config import ML_RANDOM_STATE

FAMILY = "random_forest"
SCALE_FEATURES = False

_DEFAULTS = {"n_estimators": 200, "max_depth": 6}


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    settings = {**_DEFAULTS, **params}
    model = RandomForestClassifier(
        **settings,
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
        random_state=ML_RANDOM_STATE,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def build_regressor(params: dict) -> Pipeline:
    settings = {**_DEFAULTS, **params}
    model = RandomForestRegressor(**settings, random_state=ML_RANDOM_STATE)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": None, "pca": None}
