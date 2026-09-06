"""Registro de familias de modelos.

Una familia es un módulo que expone FAMILY, SCALE_FEATURES,
build_classifier(params, scale_pos_weight), build_regressor(params) y
preprocessing_descriptor(params). El banco sólo habla con esta interfaz, así
que añadir un modelo nuevo no toca ninguna de las tres etapas.
"""

from __future__ import annotations

from types import ModuleType

from .models import (
    linear_family,
    mlp_family,
    random_forest_family,
    svm_family,
    xgboost_family,
)
from .param_filter import filter_estimator_params  # re-exported for convenience

__all__ = ["available_families", "get_family", "filter_estimator_params"]

_FAMILIES: dict[str, ModuleType] = {
    module.FAMILY: module
    for module in (
        xgboost_family,
        random_forest_family,
        linear_family,
        svm_family,
        mlp_family,
    )
}


def available_families() -> tuple[str, ...]:
    return tuple(sorted(_FAMILIES))


def get_family(name: str) -> ModuleType:
    try:
        return _FAMILIES[name]
    except KeyError:
        raise ValueError(
            f"Familia desconocida: {name!r}. Disponibles: {', '.join(available_families())}"
        ) from None
