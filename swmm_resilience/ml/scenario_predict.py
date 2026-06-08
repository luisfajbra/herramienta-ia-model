"""
Scenario predictor: given a HydrographScenario, predict per-node flood results
using pre-trained classifier and regressor artifacts (joblib format).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..extraction.dynamic_features import compute_dynamic_features
from ..extraction.static_features import extract_static_features
from ..extraction.topology import compute_topology_features
from ..ml.trainer import FEATURE_COLS
from ..validation.hydrograph_csv import HydrographScenario


def _peak_inflow_lps(series: list[tuple[float, float]]) -> float:
    """Return the maximum flow value (lps) from a node's time series."""
    if not series:
        return 0.0
    return max(v for _, v in series)


def _effective_factor(
    base_inflow_lps: float,
    peak_lps: float,
) -> float:
    """Compute the effective multiplier for this node's scenario peak vs base.

    Falls back to 1.0 when base_inflow_lps is zero to avoid division by zero.
    """
    if base_inflow_lps <= 0.0:
        return 1.0
    return peak_lps / base_inflow_lps


def predict_scenario(
    scenario: HydrographScenario,
    clf_path: Path,
    reg_path: Path,
    flood_threshold_m3: float,
    inp_path: Path,
) -> pd.DataFrame:
    """Predict per-node flooding for a HydrographScenario without running SWMM.

    Parameters
    ----------
    scenario:
        Loaded hydrograph scenario with per-node time-series data.
    clf_path:
        Path to the joblib-serialised classifier (predicts inunda 0/1).
    reg_path:
        Path to the joblib-serialised regressor (predicts log1p flood volume).
    flood_threshold_m3:
        Minimum flood volume to consider a node flooded (informational; not
        used to override the classifier output in this function).
    inp_path:
        Path to the SWMM .inp network file used to extract static topology.

    Returns
    -------
    pd.DataFrame with columns:
        node_id (str), inunda_pred (int 0/1), vol_pred_m3 (float >= 0)
    """
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)

    # --- Build static + topology features from the .inp file ---
    static_df = extract_static_features(inp_path)
    full_df = compute_topology_features(static_df, inp_path)

    # --- Restrict to nodes present in the scenario ---
    scenario_nodes = list(scenario.node_series.keys())
    mask = full_df["node_id"].astype(str).isin(set(scenario_nodes))
    full_df = full_df[mask].copy().reset_index(drop=True)

    if full_df.empty:
        return pd.DataFrame(columns=["node_id", "inunda_pred", "vol_pred_m3"])

    # --- Derive a per-node "effective factor" from scenario peak flows ---
    # Use the mean factor across all scenario nodes as the representative
    # multiplier for the dynamic feature computation, then override per-node
    # q_pico_nodo with the actual scenario peak for each node.
    peak_map: dict[str, float] = {
        nid: _peak_inflow_lps(series)
        for nid, series in scenario.node_series.items()
    }

    # Compute mean factor to use for q_pico_acum_escalado scaling
    factors = []
    for _, row in full_df.iterrows():
        nid = str(row["node_id"])
        factor = _effective_factor(
            float(row.get("base_inflow_lps", 0.0) or 0.0),
            peak_map.get(nid, 0.0),
        )
        factors.append(factor)
    mean_factor = float(np.mean(factors)) if factors else 1.0

    # Use the mean factor for dynamic features (topology-wide scaling)
    dynamic_df = compute_dynamic_features(full_df, mean_factor)

    # Override q_pico_nodo with actual scenario peak per node
    node_to_peak = full_df[["node_id"]].copy()
    node_to_peak["q_pico_nodo"] = node_to_peak["node_id"].astype(str).map(
        lambda nid: peak_map.get(nid, 0.0)
    )
    dynamic_df = dynamic_df.merge(
        node_to_peak.rename(columns={"q_pico_nodo": "_q_pico_override"}),
        on="node_id",
        how="left",
    )
    dynamic_df["q_pico_nodo"] = dynamic_df["_q_pico_override"].fillna(0.0)
    dynamic_df = dynamic_df.drop(columns=["_q_pico_override"])

    # Drop any pre-existing dynamic columns from full_df before merging, to
    # avoid pandas adding _x / _y suffixes when column names collide.
    dynamic_cols = ["factor_mult", "q_pico_nodo", "q_pico_acum_escalado"]
    static_base = full_df.drop(
        columns=[c for c in dynamic_cols if c in full_df.columns]
    )
    merged = static_base.merge(dynamic_df, on="node_id", how="left")

    X = merged[FEATURE_COLS]

    inunda_pred = clf.predict(X)
    vol_pred = np.zeros(len(X))
    flood_mask = np.asarray(inunda_pred) == 1
    if flood_mask.sum() > 0:
        vol_pred[flood_mask] = np.expm1(reg.predict(X.loc[flood_mask]))
    vol_pred = np.clip(vol_pred, a_min=0.0, a_max=None)

    result = pd.DataFrame(
        {
            "node_id": merged["node_id"].astype(str),
            "inunda_pred": pd.Series(inunda_pred, dtype=int),
            "vol_pred_m3": vol_pred,
        }
    )
    return result.reset_index(drop=True)
