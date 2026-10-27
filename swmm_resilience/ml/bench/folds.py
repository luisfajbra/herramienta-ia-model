"""Construcción de folds de validación cruzada.

Sólo se permite importar de ``sklearn.model_selection`` (splitters, que no
ajustan nada). Importar transformadores o estimadores aquí rompería la
garantía anti-fuga de la etapa 1 — ver la spec §5.1 regla 1.

``build_folds`` recibe una serie de grupos y no un dataset: eso es lo que
permitirá construir folds internos sobre el train de un fold externo cuando
se añada la búsqueda de hiperparámetros, sin modificar este módulo.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .schemas import FOLD_COLUMNS, PROTOCOLS, group_column_for

_GROUPKFOLD_SPLITS = 5


def _validate_groupkfold5_groups(groups: pd.Series) -> None:
    """Valida que hay suficientes grupos para GroupKFold5."""
    distinct = int(groups.nunique())
    if distinct < _GROUPKFOLD_SPLITS:
        raise ValueError(
            f"GroupKFold5 needs at least 5 groups; the dataset has {distinct}"
        )


def _validate_leave_one_network_out_groups(groups: pd.Series) -> None:
    if groups.isna().any():
        raise ValueError(
            "LeaveOneNetworkOut does not allow null network_id values"
        )
    distinct = int(groups.nunique(dropna=True))
    if distinct < 2:
        raise ValueError(
            "LeaveOneNetworkOut needs at least 2 networks; "
            f"the dataset has {distinct}"
        )


def n_folds_for(groups: pd.Series, protocol: str) -> int:
    """Número de folds que producirá ``protocol`` sobre ``groups``."""
    if protocol in ("LOSO", "LeaveOneShapeOut"):
        return int(groups.nunique())
    if protocol == "LeaveOneNetworkOut":
        _validate_leave_one_network_out_groups(groups)
        return int(groups.nunique())
    if protocol == "GroupKFold5":
        _validate_groupkfold5_groups(groups)
        return _GROUPKFOLD_SPLITS
    raise ValueError(f"Unknown protocol: {protocol!r}. Options: {PROTOCOLS}")


def _splitter(groups: pd.Series, protocol: str):
    if protocol in ("LOSO", "LeaveOneShapeOut"):
        return LeaveOneGroupOut()
    if protocol == "LeaveOneNetworkOut":
        _validate_leave_one_network_out_groups(groups)
        return LeaveOneGroupOut()
    if protocol == "GroupKFold5":
        _validate_groupkfold5_groups(groups)
        return GroupKFold(n_splits=_GROUPKFOLD_SPLITS)
    raise ValueError(f"Unknown protocol: {protocol!r}. Options: {PROTOCOLS}")


def build_folds(groups: pd.Series, protocol: str) -> pd.DataFrame:
    """Materializa los folds de ``protocol`` como un frame en formato largo.

    Devuelve una fila por (fold, muestra) con ``split`` en {'train','test'}.
    Persistirlo así —y no como un objeto de sklearn— es lo que permite que un
    modelo entrenado en otro script, o meses después, use exactamente los
    mismos folds.
    """
    group_column = group_column_for(protocol)
    if groups.name != group_column:
        raise ValueError(
            f"{protocol} requires groups from the column {group_column!r}; "
            f"received {groups.name!r}"
        )
    splitter = _splitter(groups, protocol)
    x_dummy = np.zeros((len(groups), 1))
    records: list[dict] = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(x_dummy, None, groups.values)
    ):
        held_out_groups = json.dumps(
            sorted(str(value) for value in groups.iloc[test_idx].unique()),
            ensure_ascii=False,
        )
        for split_name, indices in (("train", train_idx), ("test", test_idx)):
            for sample_idx in sorted(int(value) for value in indices):
                records.append(
                    {
                        "protocol": protocol,
                        "fold_id": fold_id,
                        "sample_idx": sample_idx,
                        "split": split_name,
                        "group_column": group_column,
                        "held_out_groups": held_out_groups,
                    }
                )
    return pd.DataFrame.from_records(records, columns=list(FOLD_COLUMNS))


def build_all_folds(keys: pd.DataFrame, protocols: Sequence[str]) -> pd.DataFrame:
    """Concatena los folds de varios protocolos en un solo frame."""
    return pd.concat(
        [build_folds(keys[group_column_for(protocol)], protocol) for protocol in protocols],
        ignore_index=True,
    )
