"""Informe de calidad del dataset preparado.

Restricción: sólo pandas/numpy. Ver tests/ml/bench/test_stage1_imports.py.

El informe no bloquea la ejecución salvo en tres casos que hacen imposible
entrenar de forma honesta: filas duplicadas por (run_id, node_id), cero filas
inundadas, y una feature nullable enteramente nula (ver
_raise_if_any_feature_is_entirely_null). Todo lo demás se reporta para que
quede registro, no para frenar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import FEATURE_COLUMNS_V17, NULLABLE_FEATURE_COLUMNS_V17
from ...extraction.labels import flood_label


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
    feature_columns: tuple = FEATURE_COLUMNS_V17,
    nullable_columns: frozenset = NULLABLE_FEATURE_COLUMNS_V17,
) -> dict:
    """Resume la calidad del dataset y aborta ante los dos defectos fatales.

    ``feature_columns``/``nullable_columns`` default al contrato v17 (legado);
    ``prepare_dataset`` los pasa explícitamente para que el informe refleje el
    contrato (v17 o v4_21) con el que se preparó ``X``.
    """
    duplicated = keys.duplicated(subset=["run_id", "node_id"]).sum()
    if duplicated:
        raise DatasetQualityError(
            f"The dataset has {duplicated} duplicated row(s) per (run_id, node_id). "
            "This usually means the database holds more than one snapshot; repopulate "
            "the database from scratch before training."
        )

    n_flooded = int((y_clf == 1).sum())
    if n_flooded == 0:
        raise DatasetQualityError(
            "The dataset has no flooded row (inunda == 1); the regressor "
            "cannot be trained. Check the flood threshold or the factor range."
        )

    nulls = {column: int(X[column].isna().sum()) for column in feature_columns}

    all_null_features = [
        column
        for column in feature_columns
        if column in nullable_columns and nulls[column] == len(X)
    ]
    if all_null_features:
        raise DatasetQualityError(
            f"The column(s) {all_null_features} are entirely null. The "
            "contract allows them to be nullable, but SimpleImputer(strategy='median') "
            "with keep_empty_features=False by default silently drops them, and "
            "the model would train with fewer features than its metadata "
            "(ordered_features) declares. This is not a decision the bench should "
            "make on its own: fix the data extraction, or if the emptiness is "
            "legitimate, deliberately remove the feature from the contract."
        )

    unexpected = [
        column
        for column, count in nulls.items()
        if count and column not in nullable_columns
    ]
    constant = [
        column for column in feature_columns if X[column].dropna().nunique() <= 1
    ]

    factor_labels = keys["factor_mult"].astype(str)
    flooded_by_factor = (
        y_clf.groupby(factor_labels.values).sum().astype(int).to_dict()
    )

    expected_labels = flood_label(y_reg, flood_threshold_m3)
    disagreements = int((expected_labels != y_clf.astype(int)).sum())

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
        "nullable_columns": sorted(nullable_columns),
        "unexpected_null_columns": sorted(unexpected),
        "constant_columns": constant,
        "feature_stats": {column: _stats(X[column]) for column in feature_columns},
        "threshold_disagreements": disagreements,
        "flood_threshold_m3": float(flood_threshold_m3),
    }
