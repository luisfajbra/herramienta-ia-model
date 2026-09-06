"""Registro de familias de modelos.

Una familia es un módulo que expone FAMILY, SCALE_FEATURES,
build_classifier(params, scale_pos_weight), build_regressor(params) y
preprocessing_descriptor(params). El banco sólo habla con esta interfaz, así
que añadir un modelo nuevo no toca ninguna de las tres etapas.
"""

from __future__ import annotations

from types import ModuleType

from .models import xgboost_family

_FAMILIES: dict[str, ModuleType] = {
    xgboost_family.FAMILY: xgboost_family,
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
