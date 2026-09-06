"""Informe de calidad del dataset preparado.

Restricción: sólo pandas/numpy. Ver tests/ml/bench/test_stage1_imports.py.

El informe no bloquea la ejecución salvo en dos casos que hacen imposible
entrenar: filas duplicadas por (run_id, node_id) y cero filas inundadas.
Todo lo demás se reporta para que quede registro, no para frenar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import FEATURE_COLUMNS_V17, NULLABLE_FEATURE_COLUMNS_V17


class DatasetQualityError(ValueError):
    """El dataset tiene un defecto que hace imposible entrenar sobre él."""


def _stats(series: pd.Series) -> dict:
    clean = series.dropna()
    if clean.empty:
        return {key: None for key in ("min", "p1", "p25", "p50", "p75", "p99", "max")}
    return {
        "min": float(clean.min()),
        "p1": float(np.percentile(clean, 1)),
        "p25": float(np.percentile(clean, 25)),
        "p50": float(np.percentile(clean, 50)),
        "p75": float(np.percentile(clean, 75)),
        "p99": float(np.percentile(clean, 99)),
        "max": float(clean.max()),
    }


def build_quality_report(
    keys: pd.DataFrame,
    X: pd.DataFrame,
    y_clf: pd.Series,
    y_reg: pd.Series,
    flood_threshold_m3: float,
) -> dict:
    """Resume la calidad del dataset y aborta ante los dos defectos fatales."""
    duplicated = keys.duplicated(subset=["run_id", "node_id"]).sum()
    if duplicated:
        raise DatasetQualityError(
            f"El dataset tiene {duplicated} fila(s) duplicadas por (run_id, node_id). "
            "Suele indicar que la base guarda más de un snapshot; repuebla la base "
            "desde cero antes de entrenar."
        )

    n_flooded = int((y_clf == 1).sum())
    if n_flooded == 0:
        raise DatasetQualityError(
            "El dataset no tiene ninguna fila inundada (inunda == 1); el regresor "
            "no se puede entrenar. Revisa el umbral de inundación o el rango de factores."
        )

    nulls = {column: int(X[column].isna().sum()) for column in FEATURE_COLUMNS_V17}
    unexpected = [
        column
        for column, count in nulls.items()
        if count and column not in NULLABLE_FEATURE_COLUMNS_V17
    ]
    constant = [
        column for column in FEATURE_COLUMNS_V17 if X[column].dropna().nunique() <= 1
    ]

    factor_labels = keys["factor_mult"].astype(str)
    flooded_by_factor = (
        y_clf.groupby(factor_labels.values).sum().astype(int).to_dict()
    )

    disagreements = int(
        (((y_reg > flood_threshold_m3).astype(int)) != y_clf.astype(int)).sum()
    )

    return {
        "n_rows": int(len(keys)),
        "n_runs": int(keys["run_id"].nunique()),
        "n_nodes": int(keys["node_id"].nunique()),
        "class_balance": {
            "n_flooded": n_flooded,
            "n_not_flooded": int((y_clf == 0).sum()),
            "flooded_ratio": float(n_flooded / len(y_clf)),
        },
        "rows_by_factor": factor_labels.value_counts().sort_index().to_dict(),
        "rows_by_shape": keys["shape_id"].value_counts().sort_index().to_dict(),
        "flooded_by_factor": {str(k): int(v) for k, v in sorted(flooded_by_factor.items())},
        "nulls_by_column": nulls,
        "nullable_columns": sorted(NULLABLE_FEATURE_COLUMNS_V17),
        "unexpected_null_columns": sorted(unexpected),
        "constant_columns": constant,
        "feature_stats": {column: _stats(X[column]) for column in FEATURE_COLUMNS_V17},
        "threshold_disagreements": disagreements,
        "flood_threshold_m3": float(flood_threshold_m3),
    }
