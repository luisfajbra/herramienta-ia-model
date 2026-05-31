"""
swmm_api-based I/O helpers for structured .inp manipulation and result extraction.

Responsibilities:
- Read .inp files with swmm_api (Fase 1)
- Detect nodes with inflows and their timeseries (Fase 1)
- Scale embedded hydrographs and write temporary .inp (Fase 2)
- Read flooding summary from .rpt (Fase 3)
- Read timeseries from .out for future temporal dataset (Fase 4)

PySWMM remains in runner.py for executing simulations and extracting
link/conduit statistics while equivalence with .rpt/swmm_api is being validated.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Optional

from ..config import SCENARIO_MODE_STEADY, SCENARIO_MODE_TIMESERIES

try:
    import pandas as pd
    from swmm_api import read_inp_file, read_rpt_file, read_out_file
    _SWMM_API_AVAILABLE = True
except ImportError:
    _SWMM_API_AVAILABLE = False


def _require_swmm_api():
    if not _SWMM_API_AVAILABLE:
        raise ImportError(
            "swmm-api is required for this function. Install it with: pip install swmm-api"
        )


# ---------------------------------------------------------------------------
# Fase 1 — Structured .inp reading
# ---------------------------------------------------------------------------

def load_inp(inp_file: Path | str):
    """Load a SWMM .inp file and return the SwmmInput object."""
    _require_swmm_api()
    return read_inp_file(str(inp_file), encoding="utf-8")


def list_inflow_nodes(inp) -> set[str]:
    """Return node IDs that have FLOW-type inflows defined in [INFLOWS]."""
    if "INFLOWS" not in inp:
        return set()
    return {
        node
        for (node, constituent) in inp["INFLOWS"].keys()
        if str(constituent).upper() == "FLOW"
    }


def _iter_flow_inflows(inp):
    """Yield (node_id, inflow) pairs for FLOW entries in [INFLOWS]."""
    if "INFLOWS" not in inp:
        return
    for (node, constituent), inflow in inp["INFLOWS"].items():
        if str(constituent).upper() == "FLOW":
            yield str(node), inflow


def _normalize_timeseries_name(raw_name: object) -> str | None:
    """Return a clean timeseries name or None when the inflow is steady-only."""
    if raw_name is None:
        return None
    normalized = str(raw_name).strip()
    if not normalized or normalized in {'""', "''"}:
        return None
    return normalized


def get_node_timeseries_map(inp) -> dict[str, str]:
    """Return {node_id: timeseries_name} for all FLOW inflow nodes."""
    if "INFLOWS" not in inp:
        return {}
    result = {}
    for node, inflow in _iter_flow_inflows(inp):
        ts_name = _normalize_timeseries_name(inflow.time_series)
        if ts_name is not None:
            result[node] = ts_name
    return result


def get_node_inflow_profiles(inp) -> dict[str, dict]:
    """Return node inflow profiles compatible with runner.py's format.

    Each profile is:
        {
            "timeseries": str,
            "points": [(time_min, flow_lps), ...],
            "mfactor": float,
            "baseline": float,
        }

    swmm_api stores timeseries times in hours; this function converts to minutes.
    """
    if "INFLOWS" not in inp:
        return {}

    ts_map: dict = {}
    if "TIMESERIES" in inp:
        ts_map = dict(inp["TIMESERIES"])

    profiles: dict[str, dict] = {}
    for node, inflow in _iter_flow_inflows(inp):
        ts_name = _normalize_timeseries_name(inflow.time_series)
        ts_obj = ts_map.get(ts_name)
        points: list[tuple[float, float]] = []
        if ts_obj is not None:
            points = sorted(
                (t * 60.0, v) for t, v in ts_obj.data
            )
        mfactor = inflow.scale_factor if inflow.scale_factor is not None else 1.0
        baseline = inflow.base_value if inflow.base_value is not None else 0.0
        profiles[node] = {
            "timeseries": ts_name,
            "points": points,
            "mfactor": float(mfactor),
            "baseline": float(baseline),
        }
    return profiles


def get_base_node_inflows_lps(inp) -> dict[str, float]:
    """Return {node_id: baseline_flow_lps} for nodes with FLOW inflows."""
    profiles = get_node_inflow_profiles(inp)
    return {node_id: profile["baseline"] for node_id, profile in profiles.items()}


# ---------------------------------------------------------------------------
# Fase 2 — Hydrograph scaling
# ---------------------------------------------------------------------------


def _selected_flow_inflows(inp, target_nodes: set[str] | None) -> list[tuple[str, object]]:
    """Return the FLOW inflows affected by the current scenario."""
    selected: list[tuple[str, object]] = []
    for node_id, inflow in _iter_flow_inflows(inp):
        if target_nodes is None or node_id in target_nodes:
            selected.append((node_id, inflow))
    return selected


def _clone_timeseries_object(ts_obj, name: str, scaled_data):
    """Return a timeseries object with the same metadata and replaced data."""
    cloned = copy.copy(ts_obj)
    cloned.name = name
    cloned.data = list(scaled_data)
    return cloned


def _unique_timeseries_name(inp, base_name: str, node_id: str) -> str:
    """Return a timeseries name that does not collide with existing entries."""
    candidate = f"{base_name}__scaled_{node_id}"
    index = 1
    while candidate in inp["TIMESERIES"]:
        index += 1
        candidate = f"{base_name}__scaled_{node_id}_{index}"
    return candidate


def _scale_target_timeseries(inp, selected_inflows: list[tuple[str, object]], multiplier: float) -> int:
    """Scale timeseries values used by selected nodes without mutating shared users."""
    if "TIMESERIES" not in inp:
        raise ValueError(
            "El escenario 'timeseries' requiere una seccion [TIMESERIES] en el archivo .inp."
        )

    changed_values = 0
    missing_series: list[str] = []

    for node_id, inflow in selected_inflows:
        ts_name = _normalize_timeseries_name(inflow.time_series)
        if ts_name is None:
            continue
        if ts_name not in inp["TIMESERIES"]:
            missing_series.append(ts_name)
            continue

        ts_obj = inp["TIMESERIES"][ts_name]
        scaled_data = []
        for time_value, flow_value in ts_obj.data:
            scaled_value = flow_value * multiplier
            if scaled_value != flow_value:
                changed_values += 1
            scaled_data.append((time_value, scaled_value))

        new_name = _unique_timeseries_name(inp, str(ts_name), str(node_id))
        inp["TIMESERIES"][new_name] = _clone_timeseries_object(
            ts_obj,
            new_name,
            scaled_data,
        )
        inflow.time_series = new_name

    if missing_series:
        missing = ", ".join(sorted(set(missing_series)))
        raise ValueError(
            "El escenario 'timeseries' referencia series que no existen en [TIMESERIES]: "
            f"{missing}"
        )
    if changed_values == 0:
        raise ValueError(
            "El escenario 'timeseries' no encontro valores de serie temporal para escalar. "
            "Usa el modo 'steady' si el caudal esta en Baseline dentro de [INFLOWS]."
        )
    return changed_values


def _scale_target_baselines(selected_inflows: list[tuple[str, object]], multiplier: float) -> int:
    """Scale steady-flow baselines in [INFLOWS] for the selected nodes."""
    changed_values = 0
    for _, inflow in selected_inflows:
        base_value = inflow.base_value if inflow.base_value is not None else 0.0
        scaled_value = base_value * multiplier
        if scaled_value != base_value:
            changed_values += 1
        inflow.base_value = float(scaled_value)
    return changed_values


def write_scaled_inp(
    inp_file: Path | str,
    multiplier: float,
    target_nodes: set[str] | None,
    output_file: Path | str,
    scenario_mode: str = SCENARIO_MODE_TIMESERIES,
) -> Path:
    """Write a scaled copy of inp_file according to the selected scenario mode."""
    _require_swmm_api()
    inp = load_inp(inp_file)
    output_path = Path(output_file)
    selected_inflows = _selected_flow_inflows(inp, target_nodes)

    if target_nodes is not None and not selected_inflows:
        selected = ", ".join(sorted(target_nodes))
        raise ValueError(
            f"No se encontraron inflows FLOW para los nodos seleccionados: {selected}"
        )

    if multiplier != 1.0:
        if scenario_mode == SCENARIO_MODE_TIMESERIES:
            changed_values = _scale_target_timeseries(inp, selected_inflows, multiplier)
        elif scenario_mode == SCENARIO_MODE_STEADY:
            changed_values = _scale_target_baselines(selected_inflows, multiplier)
        else:
            raise ValueError(f"Modo de escenario no soportado: {scenario_mode}")

        if changed_values == 0:
            raise ValueError(
                "El multiplicador no modifico ningun inflow del .inp. "
                "Revisa si elegiste el modo correcto entre 'timeseries' y 'steady'."
            )

    inp.write_file(str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Fase 3 — Read flooding summary from .rpt
# ---------------------------------------------------------------------------

def read_node_flooding_summary(rpt_file: Path | str) -> Optional["pd.DataFrame"]:
    """Parse the node flooding summary table from a SWMM .rpt file.

    Returns a DataFrame with columns:
        node_id, flooding_volume_m3, flooding_duration_min
    Note: .rpt reports flood volume as 10^6 litres; this function converts it
    to m3 so downstream code stores total_flood_volume_m3.

    Returns None on failure; caller should fall back to PySWMM statistics.
    """
    _require_swmm_api()
    try:
        rpt = read_rpt_file(str(rpt_file))
        node_flooding = rpt.node_flooding_summary
        if node_flooding is None or node_flooding.empty:
            return None

        df = node_flooding.reset_index()
        rename_map: dict[str, str] = {}

        col_lower = {c.lower(): c for c in df.columns}
        node_col = col_lower.get("node") or col_lower.get("name") or df.columns[0]

        for col in df.columns:
            cl = col.lower()
            if "volume" in cl and ("10^6" in cl or "vol" in cl or "m3" in cl):
                rename_map[col] = "flooding_volume_m3_raw"
            elif "duration" in cl or "hours" in cl:
                rename_map[col] = "flooding_duration_hrs"

        df = df.rename(columns=rename_map)
        df = df.rename(columns={node_col: "node_id"})

        if "flooding_volume_m3_raw" in df.columns:
            # .rpt reports volume in 10^6 litres; 1 × 10^6 L = 1000 m³
            df["flooding_volume_m3"] = pd.to_numeric(
                df["flooding_volume_m3_raw"], errors="coerce"
            ) * 1000.0
        if "flooding_duration_hrs" in df.columns:
            df["flooding_duration_min"] = pd.to_numeric(
                df["flooding_duration_hrs"], errors="coerce"
            ) * 60.0

        keep = ["node_id"]
        for col in ("flooding_volume_m3", "flooding_duration_min"):
            if col in df.columns:
                keep.append(col)
        return df[keep]

    except Exception as exc:
        warnings.warn(
            f"No se pudo leer el resumen de inundacion del .rpt '{rpt_file}': {exc}",
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Fase 4 — Read timeseries from .out (scaffold for temporal dataset)
# ---------------------------------------------------------------------------

def read_out_timeseries(out_file: Path | str):
    """Load the SWMM binary output file and return a SwmmOutput object."""
    _require_swmm_api()
    return read_out_file(str(out_file))


def get_node_series(out, node_id: str) -> "pd.DataFrame":
    """Extract per-timestep node results from a SwmmOutput object.

    Returns a DataFrame with columns:
        time_sec, total_inflow_lps, lateral_inflow_lps, depth_m,
        flooding_lps, total_outflow_lps, head_m
    """
    _require_swmm_api()
    try:
        df = out.get_part("node", node_id)
    except Exception as exc:
        raise KeyError(f"Nodo '{node_id}' no encontrado en el .out: {exc}") from exc

    df = df.reset_index()
    df = df.rename(columns=str.lower)

    time_col = next(
        (c for c in df.columns if "date" in c or "time" in c or "index" in c), None
    )
    if time_col and hasattr(df[time_col].iloc[0], "total_seconds"):
        df["time_sec"] = df[time_col].apply(lambda t: t.total_seconds())
    elif time_col:
        df["time_sec"] = pd.to_numeric(df[time_col], errors="coerce")

    column_aliases = {
        "total_inflow": "total_inflow_lps",
        "lateral_inflow": "lateral_inflow_lps",
        "depth": "depth_m",
        "flooding": "flooding_lps",
        "total_outflow": "total_outflow_lps",
        "hydraulic_head": "head_m",
        "head": "head_m",
    }
    df = df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns})
    return df


def get_link_series(out, link_id: str) -> "pd.DataFrame":
    """Extract per-timestep link results from a SwmmOutput object."""
    _require_swmm_api()
    try:
        df = out.get_part("link", link_id)
    except Exception as exc:
        raise KeyError(f"Link '{link_id}' no encontrado en el .out: {exc}") from exc

    df = df.reset_index()
    df = df.rename(columns=str.lower)

    time_col = next(
        (c for c in df.columns if "date" in c or "time" in c or "index" in c), None
    )
    if time_col and hasattr(df[time_col].iloc[0], "total_seconds"):
        df["time_sec"] = df[time_col].apply(lambda t: t.total_seconds())
    elif time_col:
        df["time_sec"] = pd.to_numeric(df[time_col], errors="coerce")

    column_aliases = {
        "flow": "flow_lps",
        "velocity": "velocity_mps",
        "depth": "depth_m",
        "capacity": "capacity_ratio",
    }
    df = df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns})
    return df
