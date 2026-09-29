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
