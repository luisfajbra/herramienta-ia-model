"""Filtro de hiperparámetros compartido por las cinco familias de modelos.

Antes de esta función, cada familia decidía a su manera qué claves de
`params` no debían llegar al estimador (`scale_pos_weight` sale de aquí y
entra a build_classifier como argumento propio, así que reenviarlo también
dentro de `params` duplicaría el keyword): xgboost lo hacía con `.pop()`,
mlp con una comprensión inline, linear excluía `alpha`/`C` en cada builder,
svm excluía `epsilon` sólo en el clasificador, y random_forest no filtraba
nada. Esa divergencia significaba que el mismo config (p.ej. el ejemplo de la
§11 del diseño, que pone `scale_pos_weight: "auto"` en un bloque de familia)
fallaba con un TypeError críptico según a qué familia se copiara.

Vive en un módulo aparte (no en registry.py) para evitar un import circular:
registry.py importa los módulos de familia, así que los módulos de familia no
pueden importar de vuelta desde registry.py.
"""

from __future__ import annotations

from typing import Collection

# Claves que toda familia recibe aparte de `params`. scale_pos_weight llega
# como argumento propio de build_classifier (o, en mlp, como kwarg explícito
# del estimador) y nunca debe reenviarse dentro de params.
RESERVED_PARAM_KEYS = frozenset({"scale_pos_weight"})


def filter_estimator_params(params: dict, *, exclude: Collection[str] = ()) -> dict:
    """Quita las claves reservadas (y las que pase `exclude`) de `params`.

    Deliberadamente NO es un allow-list: todo lo que no se excluye explícita
    pasa intacto al estimador, así que una clave desconocida (o un typo) llega
    a su __init__ y lo hace fallar con TypeError en vez de desaparecer en
    silencio -- ese defecto se corrigió una tarea atrás en linear_family y no
    debe volver aquí. `exclude` es para las exclusiones propias de cada
    familia y tarea (p.ej. `alpha` no le corresponde al clasificador lineal,
    `epsilon` no le corresponde al SVC): una clave classifier-only nunca debe
    alcanzar al regresor ni viceversa.
    """
    drop = RESERVED_PARAM_KEYS | set(exclude)
    return {key: value for key, value in params.items() if key not in drop}
