"""Write a temporary SWMM .inp file with replaced [TIMESERIES] data.

The base .inp is never modified on disk. This module takes a
HydrographScenario (loaded from a validated CSV) and produces a
scenario-specific .inp file in the supplied output directory, adjusting
END_TIME / END_DATE to match the last CSV data point.
"""

from __future__ import annotations

import datetime
import math
import warnings
from pathlib import Path

from .swmm_api_io import load_inp, get_node_timeseries_map
from ..validation.hydrograph_csv import HydrographScenario


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
            f"CSV nodes without a series reference in [INFLOWS]: {sorted(missing)}"
        )

    seen: dict[str, str] = {}  # ts_name → first node_id that references it
    for node_id in scenario.node_series:
        ts_name = node_ts_map[node_id]
        if ts_name in seen:
            raise ValueError(
                f"The series '{ts_name}' is shared by the nodes "
                f"'{seen[ts_name]}' and '{node_id}' — non-unique relation"
            )
        seen[ts_name] = node_id

    return {n: node_ts_map[n] for n in scenario.node_series}


def write_scenario_inp(
    base_inp_path: Path,
    scenario: HydrographScenario,
    out_dir: Path,
    drain_down_hours: float = 6.0,
) -> Path:
    """Write a temporary .inp with replaced [TIMESERIES] for *scenario*.

    The function:
    1. Loads the base .inp (never writes back to it).
    2. Validates a 1-to-1 mapping between scenario nodes and timeseries.
    3. Replaces each timeseries' data in-memory with the scenario values,
       appending a zero-flow point at last_time + drain_down_hours so the
       network can drain after the event (truncated simulations under-report
       flood volumes in the Node Flooding Summary).
    4. Adjusts END_TIME / END_DATE to last CSV timestamp + drain_down_hours.
    5. Writes the modified .inp to ``out_dir / <scenario_id>.inp``.

    Emits a warning listing nodes whose final CSV value exceeds 1% of their
    peak (the hydrograph does not include the recession limb).

    Parameters
    ----------
    base_inp_path:
        Path to the (unmodified) base SWMM .inp file.
    scenario:
        Validated hydrograph scenario from ``load_scenario``.
    out_dir:
        Directory where the output .inp file will be written (created if
        it does not exist).
    drain_down_hours:
        Extra zero-inflow simulation time after the last CSV point. Use 0.0
        to reproduce the exact CSV duration (no drain-down).

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

    # --- Warn when the hydrograph does not end near zero ---------------------
    unfinished = []
    for node_id, series in scenario.node_series.items():
        if not series:
            continue
        peak = max(v for _, v in series)
        last_value = series[-1][1]
        if peak > 0 and last_value > 0.01 * peak:
            unfinished.append(node_id)
    if unfinished:
        warnings.warn(
            "The hydrograph does not end near zero (>1% of the peak) at nodes: "
            f"{sorted(unfinished)}. The SWMM volume may be underestimated if "
            "the recession is not included.",
            stacklevel=2,
        )

    # --- Replace timeseries data --------------------------------------------
    if "TIMESERIES" not in inp:
        raise ValueError("The base .inp does not contain a [TIMESERIES] section")

    for node_id, ts_name in node_to_series.items():
        if ts_name not in inp["TIMESERIES"]:
            raise ValueError(
                f"Series '{ts_name}' referenced in [INFLOWS] but missing from [TIMESERIES]"
            )
        data = list(scenario.node_series[node_id])
        if drain_down_hours > 0:
            data.append((scenario.last_time_hours + drain_down_hours, 0.0))
        inp["TIMESERIES"][ts_name].data = data
        # CSV scenario values are absolute L/s inputs. Remove any base, scaling,
        # or pattern from the original INP so SWMM receives the same values.
        inflow = inp["INFLOWS"][(node_id, "FLOW")]
        inflow.scale_factor = 1.0
        inflow.base_value = 0.0
        # swmm-api omits NaN optional fields when serializing an INP. Assigning
        # Python None writes the literal token "NONE", which EPA SWMM rejects
        # as an undefined pattern object.
        inflow.pattern = math.nan

    # --- Adjust END_TIME / END_DATE -----------------------------------------
    opts = inp["OPTIONS"]
    start_dt = datetime.datetime.combine(opts["START_DATE"], opts["START_TIME"])
    total_hours = scenario.last_time_hours + max(drain_down_hours, 0.0)
    new_end_dt = start_dt + datetime.timedelta(hours=total_hours)
    opts["END_DATE"] = new_end_dt.date()
    opts["END_TIME"] = new_end_dt.strftime("%H:%M:%S")

    # --- Write output -------------------------------------------------------
    out_inp = out_dir / f"{scenario.scenario_id}.inp"
    inp.write_file(str(out_inp))
    return out_inp
