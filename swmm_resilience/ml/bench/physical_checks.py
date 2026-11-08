# swmm_resilience/ml/bench/physical_checks.py
"""Physical sanity checks for routed node flood volumes (diagnostic only).

Networks in this study are trees, so a node cannot flood more water than the
integrated inflow of itself and its upstream nodes; only the small initial
storage is outside that bound, hence the tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_KEYS = ["network_id", "run_id", "node_id", "shape_id", "factor_mult"]


def tributary_bound_violations(frame: pd.DataFrame, tolerance_m3: float = 1.0) -> pd.DataFrame:
    tributary = frame["volumen_tributario_integrado_m3"].astype(float)
    volume = frame["volume_routed_m3"].astype(float)
    mask = volume > tributary + tolerance_m3
    found = frame.loc[mask, _KEYS + ["volume_routed_m3", "volumen_tributario_integrado_m3"]].copy()
    found["ratio"] = volume[mask] / tributary[mask]
    return found.reset_index(drop=True)


def cap_at_tributary_volume(volume, tributary) -> np.ndarray:
    volume, tributary = np.asarray(volume, dtype=float), np.asarray(tributary, dtype=float)
    if volume.shape != tributary.shape:
        raise ValueError("volume and tributary must have the same shape")
    if not (np.isfinite(volume).all() and np.isfinite(tributary).all()):
        raise ValueError("volume and tributary must be finite")
    if (tributary < 0).any():
        raise ValueError("tributary volume cannot be negative")
    return np.minimum(volume, tributary)


def total_monotonicity_violations(frame: pd.DataFrame, value_column: str,
                                  truth_column: str = "vol_inundacion_m3",
                                  tolerance_m3: float = 1e-6) -> pd.DataFrame:
    """Factor steps where the predicted network total drops while SWMM does not."""
    totals = (frame.groupby(["network_id", "shape_id", "factor_mult"], sort=True)
              .agg(predicted=(value_column, "sum"), truth=(truth_column, "sum")).reset_index())
    grouped = totals.groupby(["network_id", "shape_id"], sort=False)
    totals["predicted_drop_m3"] = grouped["predicted"].diff()
    truth_step = grouped["truth"].diff()
    mask = (totals["predicted_drop_m3"] < -tolerance_m3) & (truth_step >= 0)
    return totals.loc[mask, ["network_id", "shape_id", "factor_mult", "predicted_drop_m3"]].reset_index(drop=True)


def node_monotonicity_violations(frame: pd.DataFrame, value_column: str,
                                 truth_column: str = "vol_inundacion_m3",
                                 tolerance_m3: float = 1.0) -> dict:
    """Count node-level factor steps where prediction drops and SWMM does not."""
    ordered = frame.sort_values(["network_id", "shape_id", "node_id", "factor_mult"])
    grouped = ordered.groupby(["network_id", "shape_id", "node_id"], sort=False)
    predicted_step = grouped[value_column].diff()
    truth_step = grouped[truth_column].diff()
    valid = predicted_step.notna()
    violations = valid & (predicted_step < -tolerance_m3) & (truth_step >= 0)
    return {"violations": int(violations.sum()), "steps": int(valid.sum())}
