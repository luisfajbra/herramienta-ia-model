"""Familia ``svm``: SVM APROXIMADO (mapeo RBF de Nystroem + SVM lineal).

El clasificador es un SVM lineal de margen (pérdida ``hinge``, optimizada con
SGD) sobre el mapeo RBF aproximado de Nystroem, con calibración de Platt
(``CalibratedClassifierCV(method="sigmoid", cv=3)``) para obtener
``predict_proba``, requerido por el evaluador para el AUC-ROC; es lo que con
kernel exacto hacía ``SVC(probability=True)``. El regresor es un SVR lineal
(pérdida ε-insensible) sobre el mismo mapeo.

El SVC/SVR exacto con kernel RBF no es viable para los ~154 000 registros del
banco (costo O(n²)-O(n³)): la corrida del 2026-10-05 acumuló más de 1.5 días
de CPU sin terminar. Nystroem fija el número de componentes y los solvers SGD
escalan linealmente con las filas. No es un SVM de kernel exacto: el nombre
interno ``svm`` se conserva sólo por compatibilidad con el registro y el
espacio de Optuna; el descriptor lo declara como ``svm_aproximado_nystroem``.

Sesgo conocido: ``class_weight`` moldea el margen, pero la calibración de
Platt (ajustada sobre folds internos SIN ponderar) elimina casi todo su efecto
sobre la probabilidad final. Por eso el umbral 0.5 de esta familia equivale
aproximadamente a la decisión de Bayes sin ponderar, mientras que XGBoost y el
MLP desplazan su umbral efectivo al subir ``scale_pos_weight``. El recall y el
F1 de esta familia no son directamente comparables con los de aquéllas; el AUC,
independiente del umbral, es la comparación de clasificador más justa.

Nota sobre la calibración: ``CalibratedClassifierCV(cv=3)`` usa un
``StratifiedKFold`` sin barajar sobre las filas de ENTRENAMIENTO únicamente
(no hay fuga de test), pero el corte interno depende del orden de las filas y
las filas de un mismo hidrograma están fuertemente correlacionadas; si el
frame de entrenamiento no está ordenado por forma, los folds de calibración
pueden ser optimistas.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ....config import ML_RANDOM_STATE
from ..param_filter import filter_estimator_params

FAMILY = "svm"
SCALE_FEATURES = True

_KERNEL_DEFAULTS = {"gamma": 0.05, "n_components": 200}
_CLASSIFIER_DEFAULTS = {"alpha": 1e-4, "max_iter": 2000, "tol": 1e-3}
_REGRESSOR_DEFAULTS = {
    "alpha": 1e-4,
    "epsilon": 0.1,
    "max_iter": 2000,
    "tol": 1e-3,
}


def _partition_settings(
    params: dict, model_defaults: dict, *, exclude: set[str] | None = None
) -> tuple[dict, dict]:
    """Separa parámetros de Nystroem de los propios del solver SGD."""
    model_settings = {
        **model_defaults,
        **filter_estimator_params(params, exclude=exclude or ()),
    }
    kernel_settings = {
        name: model_settings.pop(name, default)
        for name, default in _KERNEL_DEFAULTS.items()
    }
    return kernel_settings, model_settings


def _kernel_step(settings: dict) -> Nystroem:
    return Nystroem(kernel="rbf", random_state=ML_RANDOM_STATE, **settings)


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    kernel_settings, model_settings = _partition_settings(
        params, _CLASSIFIER_DEFAULTS, exclude={"epsilon"}
    )
    # Margen lineal (hinge) sobre el mapeo RBF; Platt da las probabilidades.
    margin_svm = SGDClassifier(
        **model_settings,
        loss="hinge",
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
        random_state=ML_RANDOM_STATE,
    )
    model = CalibratedClassifierCV(estimator=margin_svm, method="sigmoid", cv=3)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kernel_approx", _kernel_step(kernel_settings)),
            ("model", model),
        ]
    )


def build_regressor(params: dict) -> Pipeline:
    kernel_settings, model_settings = _partition_settings(params, _REGRESSOR_DEFAULTS)
    model = SGDRegressor(
        **model_settings,
        loss="epsilon_insensitive",
        random_state=ML_RANDOM_STATE,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kernel_approx", _kernel_step(kernel_settings)),
            ("model", model),
        ]
    )


def preprocessing_descriptor(params: dict) -> dict:
    """Describe el preprocesado y el tipo de modelo.

    ``train.py`` llama a esta función con los parámetros del clasificador y
    con los del regresor; sólo estos últimos llevan ``epsilon``, y así se
    distingue la pérdida declarada (``hinge`` + Platt vs ε-insensible).
    Limitación: un bloque de regresor SIN ``epsilon`` se describiría como
    clasificador; el contrato del registro sólo entrega ``params`` y en el
    flujo real el regresor siempre lleva ``epsilon`` (config y Optuna).
    """
    is_regressor = "epsilon" in params
    kernel_settings, _ = _partition_settings(
        params, _CLASSIFIER_DEFAULTS, exclude={"epsilon"}
    )
    descriptor = {
        "model_type": "svm_aproximado_nystroem",
        "imputer": "median",
        "scaler": "standard",
        "kernel_approx": {
            "method": "Nystroem",
            "kernel": "rbf",
            **kernel_settings,
        },
        "pca": None,
    }
    if is_regressor:
        descriptor["loss"] = "epsilon_insensitive"
    else:
        descriptor["loss"] = "hinge"
        descriptor["probability_calibration"] = "platt_sigmoid_cv3"
    return descriptor
