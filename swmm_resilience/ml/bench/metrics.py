"""Las tres capas de métricas del banco.

Replican literalmente las de ml/evaluator.py, incluida su asimetría de
agregación: las métricas del clasificador y las de end-to-end se PROMEDIAN
entre folds, mientras que las del regresor-oracle se calculan sobre el POOL
concatenado de todos los folds. Esa diferencia cambia los números y hoy es
invisible en el código original; aquí queda documentada y es lo que el test
de paridad comprueba.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe Efficiency."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def classifier_metrics(y_true, y_pred, y_prob) -> dict:
    """Nivel 1. ``auc_roc`` es NaN si el fold no tiene ambas clases."""
    y_true = np.asarray(y_true)
    has_both = y_true.sum() > 0 and (1 - y_true).sum() > 0
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
    }


def regressor_oracle_metrics(y_true, y_pred) -> dict:
    """Nivel 2. Filtrado con etiquetas REALES: es una cota superior optimista."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "nse": nse(y_true, y_pred),
        "log_nse": nse(np.log1p(y_true), np.log1p(y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def end_to_end_metrics(y_true_vol, y_pred_vol, y_true_clf, y_pred_clf) -> dict:
    """Nivel 3. Enrutado con etiquetas PREDICHAS: el error real del sistema."""
    y_true_vol = np.asarray(y_true_vol, dtype=float)
    y_pred_vol = np.asarray(y_pred_vol, dtype=float)
    return {
        "pct_nodos_correctos": float((np.asarray(y_pred_clf) == np.asarray(y_true_clf)).mean()),
        "rmse_vol_todos_nodos": float(np.sqrt(mean_squared_error(y_true_vol, y_pred_vol))),
        "vol_total_pred_m3": float(y_pred_vol.sum()),
        "vol_total_real_m3": float(y_true_vol.sum()),
    }


def _average(entries: list, key: str) -> float:
    values = [
        entry[key] for entry in entries if not np.isnan(entry.get(key, float("nan")))
    ]
    return float(np.mean(values)) if values else float("nan")


def mean_metrics(entries: list) -> dict:
    """Promedio por clave entre folds, ignorando NaN."""
    if not entries:
        return {}
    return {key: _average(entries, key) for key in entries[0]}


def pooled_regressor_metrics(true_parts: list, pred_parts: list) -> dict:
    """Métricas del regresor sobre el pool concatenado de todos los folds."""
    if not true_parts:
        return {}
    return regressor_oracle_metrics(
        np.concatenate(true_parts), np.concatenate(pred_parts)
    )
