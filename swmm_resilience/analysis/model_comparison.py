from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def build_comparison_df(
    swmm_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    scenario_id: str,
) -> pd.DataFrame:
    """Build the canonical per-node comparison DataFrame for one scenario.

    swmm_df must have columns: node_id, inunda_swmm, vol_swmm_m3
    pred_df must have columns: node_id, inunda_pred, vol_pred_m3

    Raises ValueError on node set mismatch or duplicate node_ids.
    """
    swmm_nodes = set(swmm_df["node_id"].astype(str))
    pred_nodes = set(pred_df["node_id"].astype(str))
    if swmm_nodes != pred_nodes:
        raise ValueError(
            f"Conjuntos de nodos distintos entre SWMM y predicción. "
            f"Solo en SWMM: {swmm_nodes - pred_nodes}. "
            f"Solo en pred: {pred_nodes - swmm_nodes}"
        )
    if swmm_df["node_id"].duplicated().any():
        raise ValueError("node_ids duplicados en swmm_df")
    if pred_df["node_id"].duplicated().any():
        raise ValueError("node_ids duplicados en pred_df")

    merged = swmm_df.merge(pred_df, on="node_id", how="inner")
    merged["scenario_id"] = scenario_id
    merged["clasificacion_correcta"] = (
        (merged["inunda_swmm"] == merged["inunda_pred"]).astype(int)
    )
    merged["error_m3"] = merged["vol_pred_m3"] - merged["vol_swmm_m3"]
    merged["abs_error_m3"] = merged["error_m3"].abs()

    return merged[
        [
            "scenario_id", "node_id",
            "inunda_swmm", "inunda_pred", "clasificacion_correcta",
            "vol_swmm_m3", "vol_pred_m3", "error_m3", "abs_error_m3",
        ]
    ].reset_index(drop=True)


def compute_classification_metrics(
    inunda_swmm: "array-like",
    inunda_pred: "array-like",
) -> dict:
    """Return TP, TN, FP, FN, accuracy, precision, recall, F1.

    Metrics without a valid denominator are stored as None, not 0.
    """
    y_true = np.asarray(inunda_swmm, dtype=int)
    y_pred = np.asarray(inunda_pred, dtype=int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    # Critical Success Index (threat score): ignores trivial true negatives,
    # standard in flood validation where most node-scenario pairs are dry.
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "csi": csi,
    }


def compute_conditional_volume_metrics(
    vol_swmm: "array-like",
    vol_pred: "array-like",
    inunda_swmm: "array-like",
) -> dict:
    """Volume metrics restricted to nodes that actually flooded in SWMM.

    Unconditional MAE/RMSE are deflated by the many (0, 0) node pairs; these
    metrics measure the case that matters. Returns None values when no node
    flooded.
    """
    y_true = np.asarray(vol_swmm, dtype=float)
    y_pred = np.asarray(vol_pred, dtype=float)
    mask = np.asarray(inunda_swmm, dtype=int) == 1

    n_flooded = int(mask.sum())
    if n_flooded == 0:
        return {"mae_flooded_m3": None, "rmse_flooded_m3": None, "n_flooded": 0}

    return {
        "mae_flooded_m3": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "rmse_flooded_m3": float(
            np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        ),
        "n_flooded": n_flooded,
    }


def compute_pr_auc(
    inunda_swmm: "array-like",
    prob_inunda: "array-like",
) -> float | None:
    """Average precision (PR-AUC) over pooled nodes. None if a single class."""
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(inunda_swmm, dtype=int)
    probs = np.asarray(prob_inunda, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, probs))


def compute_volume_metrics(
    vol_swmm: "array-like",
    vol_pred: "array-like",
) -> dict:
    """Return MAE, RMSE, total volumes, absolute and percentage total error."""
    y_true = np.asarray(vol_swmm, dtype=float)
    y_pred = np.asarray(vol_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    vol_total_swmm = float(y_true.sum())
    vol_total_pred = float(y_pred.sum())
    error_abs_total = float(abs(vol_total_pred - vol_total_swmm))
    error_pct_total = (
        float((vol_total_pred - vol_total_swmm) / vol_total_swmm * 100)
        if vol_total_swmm > 0
        else None
    )

    return {
        "mae_m3": mae,
        "rmse_m3": rmse,
        "vol_total_swmm_m3": vol_total_swmm,
        "vol_total_pred_m3": vol_total_pred,
        "error_abs_total_m3": error_abs_total,
        "error_pct_total": error_pct_total,
    }


def compute_per_node_r2(
    records: list[dict],
) -> dict[str, float | None]:
    """Compute R² per node across all records.

    Each record must have keys: node_id, vol_swmm_m3, vol_pred_m3.
    Returns {node_id: r2_or_None}.
    R² is None when fewer than 2 samples or variance of vol_swmm is zero.
    """
    from collections import defaultdict

    by_node: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rec in records:
        by_node[rec["node_id"]].append((rec["vol_swmm_m3"], rec["vol_pred_m3"]))

    result: dict[str, float | None] = {}
    for node_id, pairs in by_node.items():
        if len(pairs) < 2:
            result[node_id] = None
            continue
        y_true = np.array([p[0] for p in pairs], dtype=float)
        y_pred = np.array([p[1] for p in pairs], dtype=float)
        if np.var(y_true) == 0:
            result[node_id] = None
        else:
            result[node_id] = float(r2_score(y_true, y_pred))

    return result
