"""Construcción de folds de validación cruzada.

Sólo se permite importar de ``sklearn.model_selection`` (splitters, que no
ajustan nada). Importar transformadores o estimadores aquí rompería la
garantía anti-fuga de la etapa 1 — ver la spec §5.1 regla 1.

``build_folds`` recibe una serie de grupos y no un dataset: eso es lo que
permitirá construir folds internos sobre el train de un fold externo cuando
se añada la búsqueda de hiperparámetros, sin modificar este módulo.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .schemas import FOLD_COLUMNS, PROTOCOLS

_GROUPKFOLD_SPLITS = 5


def _validate_groupkfold5_groups(groups: pd.Series) -> None:
    """Valida que hay suficientes grupos para GroupKFold5."""
    distinct = int(groups.nunique())
    if distinct < _GROUPKFOLD_SPLITS:
        raise ValueError(
            f"GroupKFold5 necesita al menos 5 grupos; el dataset tiene {distinct}"
        )


def n_folds_for(groups: pd.Series, protocol: str) -> int:
    """Número de folds que producirá ``protocol`` sobre ``groups``."""
    if protocol == "LOSO":
        return int(groups.nunique())
    if protocol == "GroupKFold5":
        _validate_groupkfold5_groups(groups)
        return _GROUPKFOLD_SPLITS
    raise ValueError(f"Protocolo desconocido: {protocol!r}. Opciones: {PROTOCOLS}")


def _splitter(groups: pd.Series, protocol: str):
    if protocol == "LOSO":
        return LeaveOneGroupOut()
    if protocol == "GroupKFold5":
        _validate_groupkfold5_groups(groups)
        return GroupKFold(n_splits=_GROUPKFOLD_SPLITS)
    raise ValueError(f"Protocolo desconocido: {protocol!r}. Opciones: {PROTOCOLS}")


def build_folds(groups: pd.Series, protocol: str) -> pd.DataFrame:
    """Materializa los folds de ``protocol`` como un frame en formato largo.

    Devuelve una fila por (fold, muestra) con ``split`` en {'train','test'}.
    Persistirlo así —y no como un objeto de sklearn— es lo que permite que un
    modelo entrenado en otro script, o meses después, use exactamente los
    mismos folds.
    """
    splitter = _splitter(groups, protocol)
    x_dummy = np.zeros((len(groups), 1))
    records: list[dict] = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(x_dummy, None, groups.values)
    ):
        for split_name, indices in (("train", train_idx), ("test", test_idx)):
            for sample_idx in sorted(int(value) for value in indices):
                records.append(
                    {
                        "protocol": protocol,
                        "fold_id": fold_id,
                        "sample_idx": sample_idx,
                        "split": split_name,
                    }
                )
    return pd.DataFrame.from_records(records, columns=list(FOLD_COLUMNS))


def build_all_folds(groups: pd.Series, protocols: Sequence[str]) -> pd.DataFrame:
    """Concatena los folds de varios protocolos en un solo frame."""
    frames = [build_folds(groups, protocol) for protocol in protocols]
    return pd.concat(frames, ignore_index=True)
