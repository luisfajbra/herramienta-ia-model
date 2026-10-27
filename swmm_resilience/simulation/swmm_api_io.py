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
import re
import warnings
from pathlib import Path
from typing import Optional

from ..config import SCENARIO_MODE_STEADY, SCENARIO_MODE_TIMESERIES
from .effective_inflow import build_effective_inflow

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

    Values are interpreted with SWMM's FLOW rule: Base + Sfactor * series.
    Profiles hold the shared evaluator so all callers use identical interpolation
    and zero-outside-series behavior.

    swmm_api stores timeseries times in hours; this function converts to minutes.
    """
    if "INFLOWS" not in inp:
        return {}

    ts_map: dict = {}
    if "TIMESERIES" in inp:
        ts_map = dict(inp["TIMESERIES"])

    flow_units = None
    if "OPTIONS" in inp:
        flow_units = inp["OPTIONS"].get("FLOW_UNITS")
    profiles: dict[str, dict] = {}
    for node, inflow in _iter_flow_inflows(inp):
        ts_name = _normalize_timeseries_name(inflow.time_series)
        ts_obj = ts_map.get(ts_name)
        points: list[tuple[float, float]] = []
        if ts_obj is not None:
            points = sorted(
                (t * 60.0, v) for t, v in ts_obj.data
            )
        effective = build_effective_inflow(
            [(time_min / 60.0, value) for time_min, value in points],
            flow_units=flow_units,
            constituent="FLOW",
            scale_factor=getattr(inflow, "scale_factor", None),
            baseline=getattr(inflow, "base_value", None),
            pattern=getattr(inflow, "pattern", None),
            mass_unit_factor=getattr(inflow, "mass_unit_factor", None),
            timeseries_name=ts_name,
        )
        profiles[node] = {
            "timeseries": ts_name,
            "points": [(time_h * 60.0, value) for time_h, value in effective.points],
            "mfactor": 1.0,
            "baseline": effective.baseline_lps,
            "effective_inflow": effective,
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
            "The 'timeseries' scenario requires a [TIMESERIES] section in the .inp file."
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
            "The 'timeseries' scenario references series that do not exist in [TIMESERIES]: "
            f"{missing}"
        )
    if changed_values == 0:
        raise ValueError(
            "The 'timeseries' scenario found no time series values to scale. "
            "Use 'steady' mode if the flow is in Baseline within [INFLOWS]."
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
            f"No FLOW inflows found for the selected nodes: {selected}"
        )

    if multiplier != 1.0:
        if scenario_mode == SCENARIO_MODE_TIMESERIES:
            changed_values = _scale_target_timeseries(inp, selected_inflows, multiplier)
        elif scenario_mode == SCENARIO_MODE_STEADY:
            changed_values = _scale_target_baselines(selected_inflows, multiplier)
        else:
            raise ValueError(f"Unsupported scenario mode: {scenario_mode}")

        if changed_values == 0:
            raise ValueError(
                "The multiplier did not modify any inflow in the .inp. "
                "Check whether you chose the right mode between 'timeseries' and 'steady'."
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

        volume_columns = [
            col
            for col in df.columns
            if "volume" in col.lower()
            and ("flood" in col.lower() or "pond" in col.lower())
        ]
        total_volume_columns = [col for col in volume_columns if "total" in col.lower()]
        if len(total_volume_columns) == 1:
            volume_col = total_volume_columns[0]
        elif len(volume_columns) == 1 and "pond" not in volume_columns[0].lower():
            volume_col = volume_columns[0]
        else:
            raise ValueError(
                "Cannot identify one total flooding-volume column in Node Flooding Summary"
            )

        for col in df.columns:
            cl = col.lower()
            if col == volume_col:
                rename_map[col] = "flooding_volume_raw"
            elif "duration" in cl or "hours_flooded" in cl:
                rename_map[col] = "flooding_duration_hrs"

        df = df.rename(columns=rename_map)
        df = df.rename(columns={node_col: "node_id"})

        volume_label = str(volume_col).lower()
        if "10^6" in volume_label and ("ltr" in volume_label or "liter" in volume_label):
            volume_factor_to_m3 = 1000.0
        elif "10^6" in volume_label and "gal" in volume_label:
            volume_factor_to_m3 = 3785.411784
        elif "m3" in volume_label or "m^3" in volume_label:
            volume_factor_to_m3 = 1.0
        else:
            raise ValueError(f"Unsupported or ambiguous flooding-volume units: {volume_col}")
        df["flooding_volume_m3"] = pd.to_numeric(
            df["flooding_volume_raw"], errors="coerce"
        ) * volume_factor_to_m3
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
            f"Could not read the flooding summary from the .rpt '{rpt_file}': {exc}",
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
        raise KeyError(f"Node '{node_id}' not found in the .out: {exc}") from exc

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
        raise KeyError(f"Link '{link_id}' not found in the .out: {exc}") from exc

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


# ---------------------------------------------------------------------------
# Continuity error reading
# ---------------------------------------------------------------------------

_CONTINUITY_RE = re.compile(
    r"Continuity Error \(%\)\s*\.*\s*(-?\d+(?:\.\d+)?)"
)


def read_continuity_error(rpt_path: Path) -> float | None:
    """Worst (max |value|) continuity error (%) reported in the .rpt.

    A routing continuity error above the configured threshold means SWMM
    itself does not trust its mass balance for that scenario (spec Sec.7).
    """
    try:
        text = Path(rpt_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values = [float(v) for v in _CONTINUITY_RE.findall(text)]
    if not values:
        return None
    return max(values, key=abs)
