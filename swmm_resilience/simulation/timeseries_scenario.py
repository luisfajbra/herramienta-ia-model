"""Write a temporary SWMM .inp file with replaced [TIMESERIES] data.

The base .inp is never modified on disk. This module takes a
HydrographScenario (loaded from a validated CSV) and produces a
scenario-specific .inp file in the supplied output directory, adjusting
END_TIME / END_DATE so the network has time to drain after the last CSV
data point.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from .swmm_api_io import load_inp, get_node_timeseries_map
from ..validation.hydrograph_csv import HydrographScenario


def _parse_end_time(end_time_str: str) -> datetime.time:
    """Parse a SWMM OPTIONS END_TIME string (e.g. '3:00:00' or '03:00:00')."""
    parts = str(end_time_str).strip().split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = int(parts[2]) if len(parts) > 2 else 0
    return datetime.time(h % 24, m, s)


def _validate_one_to_one_mapping(inp, scenario: HydrographScenario) -> dict[str, str]:
    """Return {node_id: series_name} enforcing a strict 1-to-1 relationship.

    Raises ValueError if:
    - A scenario node has no FLOW inflow reference in [INFLOWS].
    - Two scenario nodes share the same timeseries name (would corrupt data).
    """
    node_ts_map = get_node_timeseries_map(inp)

    missing = [n for n in scenario.node_series if n not in node_ts_map]
    if missing:
        raise ValueError(
            f"Nodos del CSV sin referencia a serie en [INFLOWS]: {sorted(missing)}"
        )

    seen: dict[str, str] = {}  # ts_name → first node_id that references it
    for node_id in scenario.node_series:
        ts_name = node_ts_map[node_id]
        if ts_name in seen:
            raise ValueError(
                f"La serie '{ts_name}' está compartida por los nodos "
                f"'{seen[ts_name]}' y '{node_id}' — relación no unívoca"
            )
        seen[ts_name] = node_id

    return {n: node_ts_map[n] for n in scenario.node_series}


def write_scenario_inp(
    base_inp_path: Path,
    scenario: HydrographScenario,
    out_dir: Path,
) -> Path:
    """Write a temporary .inp with replaced [TIMESERIES] for *scenario*.

    The function:
    1. Loads the base .inp (never writes back to it).
    2. Validates a 1-to-1 mapping between scenario nodes and timeseries.
    3. Replaces each timeseries' data in-memory with the scenario values.
    4. Adjusts END_TIME / END_DATE so the simulation covers the full
       scenario duration plus the original base drainage period.
    5. Writes the modified .inp to ``out_dir / <scenario_id>.inp``.

    Parameters
    ----------
    base_inp_path:
        Path to the (unmodified) base SWMM .inp file.
    scenario:
        Validated hydrograph scenario from ``load_scenario``.
    out_dir:
        Directory where the output .inp file will be written (created if
        it does not exist).

    Returns
    -------
    Path
        Absolute path to the written .inp file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = load_inp(base_inp_path)

    # --- 1-to-1 validation --------------------------------------------------
    node_to_series = _validate_one_to_one_mapping(inp, scenario)

    # --- Replace timeseries data --------------------------------------------
    if "TIMESERIES" not in inp:
        raise ValueError("El .inp base no contiene sección [TIMESERIES]")

    for node_id, ts_name in node_to_series.items():
        if ts_name not in inp["TIMESERIES"]:
            raise ValueError(
                f"Serie '{ts_name}' referenciada en [INFLOWS] pero ausente en [TIMESERIES]"
            )
        inp["TIMESERIES"][ts_name].data = list(scenario.node_series[node_id])

    # --- Adjust END_TIME / END_DATE -----------------------------------------
    opts = inp["OPTIONS"]
    start_dt = datetime.datetime.combine(opts["START_DATE"], opts["START_TIME"])
    end_time_parsed = _parse_end_time(opts["END_TIME"])
    end_dt = datetime.datetime.combine(opts["END_DATE"], end_time_parsed)
    base_duration = end_dt - start_dt  # drainage tail already in the base .inp

    new_end_dt = (
        start_dt
        + datetime.timedelta(hours=scenario.last_time_hours)
        + base_duration
    )
    opts["END_DATE"] = new_end_dt.date()
    opts["END_TIME"] = new_end_dt.strftime("%H:%M:%S")

    # --- Write output -------------------------------------------------------
    out_inp = out_dir / f"{scenario.scenario_id}.inp"
    inp.write_file(str(out_inp))
    return out_inp
