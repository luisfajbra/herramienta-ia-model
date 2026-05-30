"""
Data loaders — convert SWMM DB results or ML predictions into the
standard DataFrame consumed by flood_map.plot_flood_map().

Standard columns
----------------
node_id              : str
peak_flooding_lps   : float  (0 for non-flooded nodes)
flooded              : int    (0 or 1)
source               : str    ('swmm' | 'ml')
inflow_multiplier    : float
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_DB_FILE, DEFAULT_MODEL_ARTIFACTS_DIR

_STANDARD_COLS = ["node_id", "peak_flooding_lps", "flooded", "source", "inflow_multiplier"]


def _standardize(df: pd.DataFrame, source: str, inflow_multiplier: float) -> pd.DataFrame:
    out = df.copy()
    out["source"] = source
    out["inflow_multiplier"] = inflow_multiplier
    out["peak_flooding_lps"] = out["peak_flooding_lps"].clip(lower=0.0)
    out["node_id"] = out["node_id"].astype(str)
    return out[_STANDARD_COLS].reset_index(drop=True)


# ── SWMM loader ───────────────────────────────────────────────────────────────

def load_from_swmm(run_id: str, db_path: Path | str = DEFAULT_DB_FILE) -> pd.DataFrame:
    """
    Load node results for a completed SWMM run from the SQLite database.

    Returns the standard DataFrame for one run_id.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT
                nr.node_id,
                COALESCE(nr.peak_flooding_lps, 0.0) AS peak_flooding_lps,
                COALESCE(nr.flooded, 0)              AS flooded,
                r.inflow_multiplier
            FROM node_results nr
            JOIN runs r ON r.run_id = nr.run_id
            WHERE nr.run_id = ?
        """
        df = pd.read_sql_query(query, conn, params=(run_id,))
    finally:
        conn.close()

    if df.empty:
        raise ValueError(f"No se encontraron resultados en la DB para run_id='{run_id}'")

    inflow_multiplier = float(df["inflow_multiplier"].iloc[0])
    return _standardize(df.drop(columns=["inflow_multiplier"]).assign(inflow_multiplier=inflow_multiplier),
                        source="swmm",
                        inflow_multiplier=inflow_multiplier)


def load_all_swmm_runs(db_path: Path | str = DEFAULT_DB_FILE) -> list[tuple[str, float, pd.DataFrame]]:
    """
    Load every completed SWMM run from the DB.

    Returns
    -------
    list of (run_id, inflow_multiplier, standard_DataFrame)
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        runs = pd.read_sql_query(
            "SELECT run_id, inflow_multiplier FROM runs WHERE status='completed' ORDER BY inflow_multiplier",
            conn,
        )
    finally:
        conn.close()

    results = []
    for _, row in runs.iterrows():
        df = load_from_swmm(row["run_id"], db_path)
        results.append((str(row["run_id"]), float(row["inflow_multiplier"]), df))
    return results


# ── ML loader ─────────────────────────────────────────────────────────────────

def load_from_ml(
    inp_path: Path | str,
    inflow_multiplier: float,
    artifacts_dir: Path | str = DEFAULT_MODEL_ARTIFACTS_DIR,
) -> pd.DataFrame:
    """
    Run ML inference for a single inflow_multiplier and return the standard DataFrame.
    """
    from ..ml.predict_from_inp import predict_steady_flows_from_inp  # lazy: pyswmm optional
    result = predict_steady_flows_from_inp(
        inflow_multipliers=[inflow_multiplier],
        inp_file=inp_path,
        artifacts_dir=artifacts_dir,
    )
    preds = result.predictions.copy()
    preds = preds.rename(columns={
        "predicted_peak_flooding_lps": "peak_flooding_lps",
        "predicted_flooded": "flooded",
    })
    return _standardize(preds, source="ml", inflow_multiplier=inflow_multiplier)


def load_all_ml(
    inp_path: Path | str,
    inflow_multipliers: list[float],
    artifacts_dir: Path | str = DEFAULT_MODEL_ARTIFACTS_DIR,
) -> list[tuple[float, pd.DataFrame]]:
    """
    Run ML inference for multiple multipliers.

    Returns list of (inflow_multiplier, standard_DataFrame).
    """
    from ..ml.predict_from_inp import predict_steady_flows_from_inp  # lazy: pyswmm optional
    result = predict_steady_flows_from_inp(
        inflow_multipliers=inflow_multipliers,
        inp_file=inp_path,
        artifacts_dir=artifacts_dir,
    )
    preds = result.predictions.rename(columns={
        "predicted_peak_flooding_lps": "peak_flooding_lps",
        "predicted_flooded": "flooded",
    })
    out = []
    for mult in sorted(set(inflow_multipliers)):
        subset = preds[preds["inflow_multiplier"] == mult].copy()
        out.append((mult, _standardize(subset, source="ml", inflow_multiplier=mult)))
    return out
