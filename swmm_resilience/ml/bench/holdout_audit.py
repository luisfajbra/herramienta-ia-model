"""Post hoc sensitivity of a sealed holdout evaluation; never re-selects models.

RMSE here is computed on routed volumes (dry prediction -> 0 m3), the same
definition as ``rmse_m3`` in the official holdout ``scores.json``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("network_id", "run_id", "node_id", "shape_id", "factor_mult",
                    "vol_inundacion_m3", "volume_routed_m3")


def _require(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"holdout frame lacks columns: {missing}")
    if frame.empty:
        raise ValueError("holdout frame is empty")


def _squared_error(frame: pd.DataFrame) -> np.ndarray:
    return (frame["volume_routed_m3"].to_numpy(float) - frame["vol_inundacion_m3"].to_numpy(float)) ** 2


def _rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(_squared_error(frame))))


def macro_network_rmse(frame: pd.DataFrame) -> float:
    """Mean of per-network RMSE, as in the agreed primary metric."""
    _require(frame)
    return float(np.mean([_rmse(group) for _, group in frame.groupby("network_id", sort=True)]))


def rmse_by_shape_network(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame)
    total = float(_squared_error(frame).sum())
    rows = []
    for (network, shape), group in frame.groupby(["network_id", "shape_id"], sort=True):
        sse = float(_squared_error(group).sum())
        rows.append({"network_id": int(network), "shape_id": str(shape), "n_pairs": int(len(group)),
                     "rmse_m3": _rmse(group), "sse_share": sse / total if total else 0.0})
    return pd.DataFrame(rows, columns=["network_id", "shape_id", "n_pairs", "rmse_m3", "sse_share"])


def leave_one_shape_out_macro(frame: pd.DataFrame) -> pd.DataFrame:
    """Macro RMSE with all shapes, then with each shape removed (descriptive only)."""
    _require(frame)
    networks = frame["network_id"].nunique()
    rows = [{"excluded_shape": None, "macro_network_rmse_m3": macro_network_rmse(frame)}]
    for shape in sorted(frame["shape_id"].unique()):
        rest = frame[frame["shape_id"] != shape]
        if rest.empty or rest["network_id"].nunique() != networks:
            raise ValueError(f"excluding {shape!r} removes a whole network")
        rows.append({"excluded_shape": str(shape), "macro_network_rmse_m3": macro_network_rmse(rest)})
    return pd.DataFrame(rows, columns=["excluded_shape", "macro_network_rmse_m3"])


def continuity_subset(frame: pd.DataFrame, continuity_pct_by_run: pd.Series,
                      max_abs_pct: float) -> pd.DataFrame:
    """Keep runs whose |continuity error| is within ``max_abs_pct`` percent."""
    _require(frame)
    if not (isinstance(max_abs_pct, (int, float)) and max_abs_pct > 0):
        raise ValueError("max_abs_pct must be positive")
    runs = sorted(int(run) for run in frame["run_id"].unique())
    known = continuity_pct_by_run.reindex(runs)
    if known.isna().any():
        raise ValueError("continuity is missing for holdout runs")
    keep = set(known[known.abs() <= max_abs_pct].index)
    return frame[frame["run_id"].isin(keep)]
