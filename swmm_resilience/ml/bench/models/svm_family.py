"""Familia ``svm`` compatible: mapeo RBF de Nystroem y optimización SGD.

SVC/SVR con kernel RBF exacto no es viable para los ~154 000 registros del
banco. Nystroem conserva una representación no lineal tipo RBF con un número
fijo de componentes; los estimadores SGD que siguen escalan linealmente con
el número de filas. ``log_loss`` mantiene ``predict_proba``, requerido por el
evaluador para calcular AUC-ROC. Por ello el clasificador es regresión
logística con kernel aproximado, no un SVM de margen; el nombre interno se
conserva para mantener compatible el registro del banco.
"""

from __future__ import annotations

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
    model = SGDClassifier(
        **model_settings,
        loss="log_loss",
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
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
    kernel_settings, _ = _partition_settings(
        params, _CLASSIFIER_DEFAULTS, exclude={"epsilon"}
    )
    return {
        "imputer": "median",
        "scaler": "standard",
        "kernel_approx": {
            "method": "Nystroem",
            "kernel": "rbf",
            **kernel_settings,
        },
        "pca": None,
    }
