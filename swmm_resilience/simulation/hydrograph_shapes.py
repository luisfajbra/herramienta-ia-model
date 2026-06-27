"""Normalized hydrograph shape loading and per-node scenario construction."""

from __future__ import annotations

import csv
from pathlib import Path


def load_shape(csv_path: Path) -> list[tuple[float, float]]:
    """Read a shape CSV (columns: time_h, q_norm) and return [(time_h, q_norm), ...]."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row["time_h"]), float(row["q_norm"])))
    if not rows:
        raise ValueError(f"Shape CSV vacío: {csv_path}")
    if rows[0][0] != 0.0:
        raise ValueError(f"Shape CSV debe empezar en t=0: {csv_path}")
    return rows


def load_all_shapes(shapes_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """Load all *.csv files in shapes_dir. Returns {stem: shape}."""
    shapes = {}
    for csv_path in sorted(shapes_dir.glob("*.csv")):
        shapes[csv_path.stem] = load_shape(csv_path)
    return shapes


def get_shape_stats(shape: list[tuple[float, float]]) -> tuple[float, float]:
    """Return (duracion_horas, tiempo_al_pico_h) for a shape."""
    if not shape:
        return 0.0, 0.0
    duracion_horas = shape[-1][0]
    tiempo_al_pico_h = max(shape, key=lambda x: x[1])[0]
    return duracion_horas, tiempo_al_pico_h


def apply_shape(
    shape: list[tuple[float, float]],
    base_inflows: dict[str, float],
    factor: float,
) -> dict[str, list[tuple[float, float]]]:
    """Build per-node time series: flow(t) = base_inflow * factor * q_norm(t).

    Only includes nodes where base_inflow > 0 (nodes with timeseries in the .inp).
    """
    return {
        node_id: [(t, base * factor * q_norm) for t, q_norm in shape]
        for node_id, base in base_inflows.items()
        if base > 0.0
    }


def normalize_from_csv(
    csv_path: Path,
    node_col: str | None = None,
) -> list[tuple[float, float]]:
    """Derive a normalized shape profile from an absolute-flow validation CSV.

    The CSV must have a 'time_h' column and one or more node columns with
    absolute L/s values. The resulting shape has max(q_norm) == 1.0.

    If node_col is given, only that column is used to find the peak.
    Otherwise the per-row maximum across all node columns is used.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "time_h" not in df.columns:
        raise ValueError(f"CSV debe tener columna 'time_h': {csv_path}")

    time_vals = df["time_h"].values.tolist()

    if node_col:
        values = df[node_col].values.astype(float)
    else:
        node_cols = [c for c in df.columns if c != "time_h"]
        if not node_cols:
            raise ValueError(f"No hay columnas de nodos en {csv_path}")
        values = df[node_cols].max(axis=1).values.astype(float)

    peak = float(values.max())
    if peak == 0.0:
        raise ValueError(f"Pico es 0 en {csv_path} — no se puede normalizar")

    q_norm = (values / peak).tolist()
    return list(zip(time_vals, q_norm))
