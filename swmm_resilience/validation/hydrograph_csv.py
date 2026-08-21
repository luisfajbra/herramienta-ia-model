from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_TIME_RE = re.compile(r"^(\d+):([0-5]\d)$")


@dataclass
class HydrographScenario:
    scenario_id: str
    node_series: dict[str, list[tuple[float, float]]]  # node_id → [(hours, lps)]
    time_grid_hours: list[float]
    last_time_hours: float


def scenario_id_from_path(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def _parse_time_hours(s: str) -> float | None:
    m = _TIME_RE.match(str(s).strip())
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) / 60.0


def load_scenario(csv_path: Path, expected_nodes: set[str]) -> HydrographScenario:
    """Load and validate a hydrograph CSV scenario.

    Raises ValueError with a descriptive message if any validation rule fails.
    Rules 1–9 from spec section 6 (rule 10 — unique IDs across a batch —
    is enforced by the batch coordinator).
    """
    df = pd.read_csv(csv_path)

    # Rule 1: exactly the required columns
    required = {"node_id", "time", "value_lps"}
    missing_cols = required - set(df.columns)
    extra_cols = set(df.columns) - required
    if missing_cols or extra_cols:
        raise ValueError(
            f"Columnas inválidas. Faltantes: {sorted(missing_cols)}. "
            f"Adicionales: {sorted(extra_cols)}"
        )

    df["node_id"] = df["node_id"].astype(str)

    # Rule 2: exactly the expected nodes
    csv_nodes = set(df["node_id"].unique())
    if csv_nodes != expected_nodes:
        raise ValueError(
            f"Nodos incorrectos. Faltantes: {sorted(expected_nodes - csv_nodes)}. "
            f"Adicionales: {sorted(csv_nodes - expected_nodes)}"
        )

    # Rule 5: valid H:MM format
    parsed = df["time"].apply(lambda t: _parse_time_hours(str(t).strip()))
    bad = df["time"][parsed.isna()].unique().tolist()
    if bad:
        raise ValueError(f"Tiempos con formato inválido: {bad}")
    df["_th"] = parsed

    # Rule 6: no duplicate (node_id, time) keys
    if df.duplicated(subset=["node_id", "time"]).any():
        raise ValueError("Claves (node_id, time) duplicadas en el CSV")

    # Rule 9: finite non-negative values
    vals = pd.to_numeric(df["value_lps"], errors="coerce")
    if vals.isna().any():
        raise ValueError("value_lps contiene valores no numéricos o nulos")
    if not np.isfinite(vals.to_numpy()).all():
        raise ValueError("value_lps contiene valores infinitos")
    if (vals < 0).any():
        raise ValueError("value_lps contiene valores negativos")
    df["_v"] = vals

    # Rules 3, 4, 7 per node and rule 8 (shared grid)
    ref_grid: list[float] | None = None
    node_series: dict[str, list[tuple[float, float]]] = {}

    for nid, grp in df.groupby("node_id", sort=False):
        grp_sorted = grp.sort_values("_th")
        times = grp_sorted["_th"].tolist()

        # Rule 3: at least two time steps
        if len(times) < 2:
            raise ValueError(f"Nodo '{nid}' tiene menos de 2 tiempos")

        # Rule 4: start at 0:00 (checked on sorted data so row order doesn't matter)
        if times[0] != 0.0:
            raise ValueError(f"Nodo '{nid}' no comienza en 0:00")

        # Rule 7: strictly increasing (on sorted times)
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                raise ValueError(
                    f"Nodo '{nid}' tiene tiempos no estrictamente crecientes"
                )

        # Rule 8: same grid for all nodes
        if ref_grid is None:
            ref_grid = times
        elif not np.allclose(times, ref_grid):
            raise ValueError(
                f"Nodo '{nid}' tiene una malla temporal diferente al resto"
            )

        node_series[nid] = list(zip(grp_sorted["_th"], grp_sorted["_v"]))

    if ref_grid is None:
        raise ValueError("El CSV no contiene filas de datos")

    return HydrographScenario(
        scenario_id=scenario_id_from_path(csv_path),
        node_series=node_series,
        time_grid_hours=ref_grid,
        last_time_hours=ref_grid[-1],
    )


def _hours_to_hhmm(h: float) -> str:
    total_min = round(h * 60)
    return f"{total_min // 60}:{total_min % 60:02d}"


def write_shape_validation_csv(
    shape_id: str,
    shape: list[tuple[float, float]],
    base_inflows: dict[str, float],
    factor: float,
    out_dir: Path,
) -> Path:
    """Write a validation CSV for (shape_id, factor) in the expected format.

    Only nodes with base_inflow > 0 are written (same set as expected_nodes
    from the base .inp).  Decimal-hour time values are rounded to the nearest
    minute and serialised as H:MM so load_scenario can read them back.

    Returns the path of the written CSV.
    """
    from swmm_resilience.simulation.hydrograph_shapes import apply_shape

    node_series = apply_shape(shape, base_inflows, factor)
    if not node_series:
        raise ValueError(f"No nodes with positive base inflow for shape '{shape_id}'")

    rows = [
        {"node_id": nid, "time": _hours_to_hhmm(t_h), "value_lps": round(v, 6)}
        for nid, series in sorted(node_series.items())
        for t_h, v in series
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{shape_id}_f{factor:.3f}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path
