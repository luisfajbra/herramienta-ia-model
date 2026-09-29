"""Búsqueda automática de hiperparámetros (Optuna), aislada del banco.

Este módulo NUNCA se invoca desde --bench: solo desde --bench-tune,
explícito, sobre una sola familia. Los folds que genera son internos y
efímeros -- GroupKFold(5) agrupado por shape_id sobre todo el
PreparedDataset -- y nunca se declaran como protocolo del banco: no tocan
PROTOCOL_GROUP_COLUMNS, group_column_for() ni el manifiesto persistido de
PreparedDataset (ver spec docs/superpowers/specs/2026-09-29-optuna-tuning-
familia-ganadora-design.md §4.2).

Al no reservar un holdout de formas separado, todo resultado de un estudio
se etiqueta "estimacion_exploratoria_condicionada" (spec §12 del diseño de
cierre de auditoría): no debe presentarse como confirmatorio.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from ...config import ML_RANDOM_STATE


def build_tuning_folds(
    keys: pd.DataFrame, n_splits: int = 5, seed: int = ML_RANDOM_STATE
) -> pd.DataFrame:
    """GroupKFold(n_splits) agrupado por shape_id sobre todo `keys`.

    Devuelve un frame con columnas ('fold_id', 'split', 'sample_idx'), sin
    'protocol': estos folds son efímeros, por-trial, y nunca se escriben en
    el manifiesto de PreparedDataset.
    """
    groups = keys["shape_id"]
    distinct = int(groups.nunique())
    if distinct < n_splits:
        raise ValueError(
            f"GroupKFold({n_splits}) para tuning necesita al menos {n_splits} "
            f"formas (shape_id) distintas; el dataset tiene {distinct}."
        )
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x_dummy = np.zeros((len(groups), 1))
    records: list[dict] = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(x_dummy, None, groups.values)
    ):
        for split_name, indices in (("train", train_idx), ("test", test_idx)):
            for sample_idx in sorted(int(value) for value in indices):
                records.append(
                    {"fold_id": fold_id, "split": split_name, "sample_idx": sample_idx}
                )
    return pd.DataFrame.from_records(records, columns=["fold_id", "split", "sample_idx"])


# Notación por parámetro: ("int", low, high) | ("float", low, high) |
# ("float_log", low, high) muestreado log-uniforme | ("categorical", [...]).
# Rangos centrados en los defaults actuales de config.yaml (ver spec §6).
SEARCH_SPACES: dict[str, dict[str, dict[str, tuple]]] = {
    "xgboost": {
        "classifier": {
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 1e-3, 0.3),
            "n_estimators": ("int", 50, 500),
            "subsample": ("float", 0.5, 1.0),
        },
        "regressor": {
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 1e-3, 0.3),
            "n_estimators": ("int", 50, 500),
            "subsample": ("float", 0.5, 1.0),
        },
    },
    "random_forest": {
        "classifier": {"max_depth": ("int", 3, 15), "n_estimators": ("int", 50, 500)},
        "regressor": {"max_depth": ("int", 3, 15), "n_estimators": ("int", 50, 500)},
    },
    "linear": {
        "classifier": {"C": ("float_log", 1e-3, 100.0)},
        "regressor": {"alpha": ("float_log", 1e-3, 100.0)},
    },
    "svm": {
        "classifier": {
            "C": ("float_log", 1e-2, 100.0),
            "gamma": ("categorical", ["scale", "auto"]),
        },
        "regressor": {
            "C": ("float_log", 1e-2, 100.0),
            "epsilon": ("float", 0.01, 1.0),
        },
    },
    "mlp": {
        "classifier": {
            "hidden_sizes": ("categorical", [(32,), (64, 32), (128, 64), (64, 32, 16)]),
            "learning_rate": ("float_log", 1e-4, 1e-2),
            "dropout": ("float", 0.0, 0.5),
        },
        "regressor": {
            "hidden_sizes": ("categorical", [(32,), (64, 32), (128, 64), (64, 32, 16)]),
            "learning_rate": ("float_log", 1e-4, 1e-2),
            "dropout": ("float", 0.0, 0.5),
        },
    },
}


def _sample_params(trial, space: dict, *, prefix: str) -> dict:
    """Muestrea un dict de hiperparámetros desde `space` usando `trial`.

    `prefix` evita colisiones de nombre entre clasificador y regresor
    dentro del mismo trial (p.ej. "xgboost_classifier_max_depth" vs
    "xgboost_regressor_max_depth"): Optuna identifica cada valor muestreado
    por su nombre completo dentro del estudio.
    """
    params: dict = {}
    for name, spec in space.items():
        kind = spec[0]
        full_name = f"{prefix}{name}"
        if kind == "int":
            params[name] = trial.suggest_int(full_name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(full_name, spec[1], spec[2])
        elif kind == "float_log":
            params[name] = trial.suggest_float(full_name, spec[1], spec[2], log=True)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(full_name, spec[1])
        else:
            raise ValueError(f"Tipo de espacio desconocido: {kind!r}")
    return params
