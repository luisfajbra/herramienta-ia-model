# swmm_resilience/ml/bench/partition_coverage.py
"""Inflow-volume coverage between development and held-out runs.

``stratified_shape_draw`` is for designing a *future* reservation; it must not
be used to redraw a holdout that has already been scored.
"""
from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import pandas as pd

_RUN_COLUMNS = ["network_id", "run_id", "shape_id", "factor_mult", "total_inflow_m3"]


def run_inflow_volumes(features: pd.DataFrame) -> pd.DataFrame:
    """Sum each run's local integrated node inflow volumes (m3)."""
    volume = features["volumen_local_integrado_m3"].astype(float)
    if not np.isfinite(volume).all():
        raise ValueError("volumen_local_integrado_m3 must be finite")
    grouped = features.assign(volumen_local_integrado_m3=volume).groupby("run_id", sort=True)
    identity = grouped[["network_id", "shape_id", "factor_mult"]].nunique()
    if (identity != 1).any().any():
        raise ValueError("each run must have a single network, shape and factor identity")
    first = grouped[["network_id", "shape_id", "factor_mult"]].first()
    result = first.assign(total_inflow_m3=grouped["volumen_local_integrado_m3"].sum()).reset_index()
    result["network_id"] = result["network_id"].astype(int)
    result["run_id"] = result["run_id"].astype(int)
    return result.loc[:, _RUN_COLUMNS]


def inflow_coverage(run_volumes: pd.DataFrame, development_run_ids: Iterable[int],
                    holdout_run_ids: Iterable[int]) -> pd.DataFrame:
    development, holdout = {int(r) for r in development_run_ids}, {int(r) for r in holdout_run_ids}
    if development & holdout:
        raise ValueError("development and holdout runs overlap")
    known = set(run_volumes["run_id"].astype(int))
    if not (development | holdout) <= known:
        raise ValueError("unknown run ids in development or holdout")
    rows = []
    for network, group in run_volumes.groupby("network_id", sort=True):
        dev = group[group["run_id"].isin(development)]
        held = group[group["run_id"].isin(holdout)]
        if dev.empty or held.empty:
            raise ValueError(f"network {network} lacks development or holdout runs")
        dev_top, held_top = dev.loc[dev["total_inflow_m3"].idxmax()], held.loc[held["total_inflow_m3"].idxmax()]
        dev_max = float(dev_top["total_inflow_m3"])
        rows.append({"network_id": int(network), "development_max_inflow_m3": dev_max,
                     "development_max_shape": str(dev_top["shape_id"]),
                     "holdout_max_inflow_m3": float(held_top["total_inflow_m3"]),
                     "holdout_max_shape": str(held_top["shape_id"]),
                     "holdout_runs_above_development_max": int((held["total_inflow_m3"] > dev_max).sum()),
                     "max_ratio_to_development_max": float(held_top["total_inflow_m3"]) / dev_max})
    return pd.DataFrame(rows)


def shape_representative_volume(run_volumes: pd.DataFrame) -> pd.Series:
    """Max over networks/factors of a shape's inflow, normalised by its network maximum."""
    network_max = run_volumes.groupby("network_id")["total_inflow_m3"].transform("max")
    normalised = run_volumes.assign(normalised=run_volumes["total_inflow_m3"] / network_max)
    return normalised.groupby("shape_id")["normalised"].max().astype(float)


def stratified_shape_draw(shape_volumes: pd.Series, k: int, seed: int,
                          keep_top_in_development: int = 1) -> list[str]:
    """Draw one shape per volume stratum, always keeping the largest shapes in development."""
    if k < 1 or keep_top_in_development < 0:
        raise ValueError("k must be >= 1 and keep_top_in_development >= 0")
    volumes = shape_volumes.astype(float)
    if volumes.index.duplicated().any() or not np.isfinite(volumes.to_numpy()).all():
        raise ValueError("shape volumes must be unique and finite")
    ordered = sorted(volumes.index, key=lambda shape: (volumes[shape], str(shape)))
    if keep_top_in_development >= len(ordered):
        raise ValueError("keep_top_in_development must be smaller than the number of shapes")
    candidates = ordered[:len(ordered) - keep_top_in_development]
    if len(candidates) < k:
        raise ValueError("not enough candidate shapes for the requested draw")
    rng = random.Random(seed)
    strata = np.array_split(np.array(candidates, dtype=object), k)
    return sorted(str(rng.choice(list(stratum))) for stratum in strata)
