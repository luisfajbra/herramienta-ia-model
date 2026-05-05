"""
Simulation-only logic: PySWMM setup, topology extraction and hydraulic results.
"""

import csv
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from ..utils import (
    circular_full_flow_lps,
    new_id,
    node_type_str,
    safe_round,
)


def _parse_inp_sections(inp_file: str):
    """Parse the minimum conduit and xsection data needed from a SWMM .inp file."""
    conduit_rows = {}
    xsection_rows = {}
    inflow_rows = defaultdict(float)
    current_section = None

    for raw_line in Path(inp_file).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";;"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line.upper()
            continue

        parts = line.split()
        if current_section == "[CONDUITS]" and len(parts) >= 6:
            link_id = parts[0]
            conduit_rows[link_id] = {
                "inlet_node": parts[1],
                "outlet_node": parts[2],
                "length_m": float(parts[3]),
                "roughness": float(parts[4]),
                "inlet_offset": float(parts[5]),
                "outlet_offset": float(parts[6]) if len(parts) > 6 else 0.0,
            }
        elif current_section == "[XSECTIONS]" and len(parts) >= 3:
            link_id = parts[0]
            xsection_rows[link_id] = {
                "shape": parts[1],
                "geom1": float(parts[2]),
            }
        elif current_section == "[INFLOWS]" and len(parts) >= 2:
            try:
                inflow_rows[parts[0]] += float(parts[-1])
            except ValueError:
                continue

    return conduit_rows, xsection_rows, dict(inflow_rows)


def load_hydrograph(hydrograph_file):
    """Load a CSV hydrograph with time in minutes and flow in L/s."""
    if hydrograph_file is None:
        return None

    path = Path(hydrograph_file)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el hidrograma: {path}")

    time_columns = ("minute", "minutes", "time_min", "elapsed_min")
    flow_columns = ("inflow_lps", "flow_lps", "caudal_lps", "q_lps")
    points = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("El hidrograma debe tener encabezados de columna.")

        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        time_col = next((normalized[col] for col in time_columns if col in normalized), None)
        flow_col = next((normalized[col] for col in flow_columns if col in normalized), None)
        if time_col is None or flow_col is None:
            raise ValueError(
                "El hidrograma debe tener columnas 'minute' e 'inflow_lps' "
                "(tambien sirven: time_min/flow_lps/caudal_lps/q_lps)."
            )

        for row in reader:
            points.append((float(row[time_col]), float(row[flow_col])))

    points.sort(key=lambda point: point[0])
    if not points:
        raise ValueError("El hidrograma no tiene filas de datos.")
    if points[0][0] < 0:
        raise ValueError("El tiempo del hidrograma no puede ser negativo.")

    return points


def _hydrograph_value_lps(hydrograph, elapsed_min: float) -> float:
    """Linearly interpolate the hydrograph value for the current simulation minute."""
    if not hydrograph:
        return 0.0
    if elapsed_min <= hydrograph[0][0]:
        return hydrograph[0][1]

    for left, right in zip(hydrograph, hydrograph[1:]):
        left_min, left_flow = left
        right_min, right_flow = right
        if elapsed_min <= right_min:
            span = right_min - left_min
            if span <= 0:
                return right_flow
            fraction = (elapsed_min - left_min) / span
            return left_flow + fraction * (right_flow - left_flow)

    return hydrograph[-1][1]


def extract_static_topology(inp_file: str, net_hash: str, Nodes, Links, Simulation):
    """Extract static network topology and aggregated node/link features from SWMM."""
    node_records = []
    link_records = []
    link_static = {}
    conduit_rows, xsection_rows, base_node_inflows_lps = _parse_inp_sections(inp_file)

    upstream_links = defaultdict(list)
    downstream_links = defaultdict(list)

    with Simulation(inp_file) as sim:
        nodes = list(Nodes(sim))
        links = list(Links(sim))

        for link in links:
            link_id = link.linkid

            link_type = "conduit"
            try:
                if link.is_weir():
                    link_type = "weir"
            except AttributeError:
                pass
            try:
                if link.is_orifice():
                    link_type = "orifice"
            except AttributeError:
                pass
            try:
                if link.is_pump():
                    link_type = "pump"
            except AttributeError:
                pass

            conduit_data = conduit_rows.get(link_id, {})
            xsection_data = xsection_rows.get(link_id, {})

            diameter = xsection_data.get("geom1", getattr(link, "geom1", None))
            length = conduit_data.get("length_m", getattr(link, "length", None))
            roughness = conduit_data.get("roughness", getattr(link, "roughness", None))
            inlet_node = conduit_data.get("inlet_node", getattr(link, "inlet_node", None))
            outlet_node = conduit_data.get("outlet_node", getattr(link, "outlet_node", None))
            inlet_offset = conduit_data.get("inlet_offset", getattr(link, "inlet_offset", 0.0))
            outlet_offset = conduit_data.get("outlet_offset", getattr(link, "outlet_offset", 0.0))

            slope = None
            try:
                if length and length > 0:
                    in_node = [node for node in nodes if node.nodeid == inlet_node]
                    out_node = [node for node in nodes if node.nodeid == outlet_node]
                    if in_node and out_node:
                        z_in = in_node[0].invert_elevation + inlet_offset
                        z_out = out_node[0].invert_elevation + outlet_offset
                        dz = z_in - z_out
                        slope = round(dz / length, 6) if dz != 0 else 0.0
            except Exception:
                slope = None

            capacity = circular_full_flow_lps(diameter, abs(slope) if slope else None, roughness)

            link_static[link_id] = {
                "diameter_m": diameter,
                "length_m": length,
                "roughness": roughness,
                "slope_m_per_m": slope,
                "full_flow_capacity_lps": capacity,
                "inlet_node": inlet_node,
                "outlet_node": outlet_node,
                "link_type": link_type,
            }

            link_records.append({
                "link_uid": link_id,
                "network_hash": net_hash,
                "inlet_node": inlet_node,
                "outlet_node": outlet_node,
                "link_type": link_type,
                "diameter_m": safe_round(diameter),
                "length_m": safe_round(length),
                "roughness": safe_round(roughness, 6),
                "slope_m_per_m": safe_round(slope, 6),
                "full_flow_capacity_lps": safe_round(capacity, 6),
            })

            if inlet_node:
                downstream_links[inlet_node].append(link_id)
            if outlet_node:
                upstream_links[outlet_node].append(link_id)

        for node in nodes:
            node_id = node.nodeid
            node_type = node_type_str(node)

            upstream_ids = upstream_links.get(node_id, [])
            downstream_ids = downstream_links.get(node_id, [])

            def agg(link_ids: list, field: str):
                return [
                    link_static[link_id][field]
                    for link_id in link_ids
                    if link_static.get(link_id, {}).get(field) is not None
                ]

            def safe_avg(values):
                return round(sum(values) / len(values), 6) if values else None

            def safe_max(values):
                return round(max(values), 6) if values else None

            def safe_min(values):
                return round(min(values), 6) if values else None

            def safe_sum(values):
                return round(sum(values), 6) if values else None

            upstream_diams = agg(upstream_ids, "diameter_m")
            upstream_slopes = [abs(value) for value in agg(upstream_ids, "slope_m_per_m") if value is not None]
            upstream_caps = agg(upstream_ids, "full_flow_capacity_lps")

            downstream_diams = agg(downstream_ids, "diameter_m")
            downstream_slopes = [abs(value) for value in agg(downstream_ids, "slope_m_per_m") if value is not None]
            downstream_caps = agg(downstream_ids, "full_flow_capacity_lps")

            node_records.append({
                "node_uid": node_id,
                "network_hash": net_hash,
                "invert_elev_m": safe_round(node.invert_elevation),
                "full_depth_m": safe_round(node.full_depth),
                "base_inflow_lps": safe_round(base_node_inflows_lps.get(node_id, 0.0), 6),
                "node_type": node_type,
                "in_degree": len(upstream_ids),
                "out_degree": len(downstream_ids),
                "upstream_pipes_count": len(upstream_ids),
                "upstream_diam_max_m": safe_max(upstream_diams),
                "upstream_diam_min_m": safe_min(upstream_diams),
                "upstream_diam_avg_m": safe_avg(upstream_diams),
                "upstream_slope_avg": safe_avg(upstream_slopes),
                "upstream_slope_max": safe_max(upstream_slopes),
                "upstream_capacity_lps": safe_sum(upstream_caps),
                "downstream_pipes_count": len(downstream_ids),
                "downstream_diam_max_m": safe_max(downstream_diams),
                "downstream_diam_min_m": safe_min(downstream_diams),
                "downstream_diam_avg_m": safe_avg(downstream_diams),
                "downstream_slope_avg": safe_avg(downstream_slopes),
                "downstream_slope_max": safe_max(downstream_slopes),
                "downstream_capacity_lps": safe_sum(downstream_caps),
            })

    return {
        "node_records": node_records,
        "link_records": link_records,
        "link_static": link_static,
        "base_node_inflows_lps": base_node_inflows_lps,
    }


def run_simulation(
    inp_file,
    inflow_multiplier,
    link_static,
    Nodes,
    Links,
    Simulation,
    hydrograph=None,
    hydrograph_multiplier: float = 1.0,
    target_nodes=None,
    base_node_inflows_lps=None,
):
    """Execute one SWMM run with multiplicative steady inflow or hydrograph forcing."""

    first_flood_step = None
    step_count = 0
    timestep_sec = None
    depth_rate_tracker = {}
    base_node_inflows_lps = base_node_inflows_lps or {}
    if target_nodes is None:
        target_node_set = None
    elif isinstance(target_nodes, str):
        target_node_set = {target_nodes}
    else:
        target_node_set = {str(node_id) for node_id in target_nodes}

    def steady_added_inflow_lps(node_id: str) -> float:
        base_inflow_lps = base_node_inflows_lps.get(node_id, 0.0) or 0.0
        return base_inflow_lps * inflow_multiplier

    def scenario_reference_inflow_lps(node_id: str) -> float:
        if target_node_set is not None and node_id not in target_node_set:
            return 0.0
        if hydrograph is not None:
            return _hydrograph_value_lps(hydrograph, 0.0) * hydrograph_multiplier
        return steady_added_inflow_lps(node_id)

    with Simulation(inp_file) as sim:
        sim_start = sim.start_time
        nodes = list(Nodes(sim))
        links = list(Links(sim))

        for node in nodes:
            depth_rate_tracker[node.nodeid] = {"prev": 0.0, "max_rate": 0.0}

        for _ in sim:
            step_count += 1
            if timestep_sec is None:
                dt = (sim.current_time - sim_start).total_seconds()
                timestep_sec = dt if dt > 0 else 1.0

            for node in nodes:
                node_id = node.nodeid
                elapsed_min = (sim.current_time - sim_start).total_seconds() / 60.0
                applies_to_node = target_node_set is None or node_id in target_node_set
                base_inflow_lps = base_node_inflows_lps.get(node_id, 0.0) or 0.0
                inflow_lps = (
                    _hydrograph_value_lps(hydrograph, elapsed_min) * hydrograph_multiplier
                    if hydrograph is not None
                    else base_inflow_lps * inflow_multiplier
                )
                applied_lps = inflow_lps if applies_to_node else 0.0
                node.generated_inflow(applied_lps)

                if first_flood_step is None and node.flooding > 0:
                    first_flood_step = step_count

                depth = node.depth
                tracker = depth_rate_tracker[node_id]
                if timestep_sec > 0:
                    rate = (depth - tracker["prev"]) / (timestep_sec / 60.0)
                    if rate > tracker["max_rate"]:
                        tracker["max_rate"] = rate
                tracker["prev"] = depth

        node_records = []
        run_inputs = []
        total_flooded = 0

        for node in nodes:
            node_id = node.nodeid
            stats = node.statistics

            full_depth = node.full_depth
            max_depth = stats.get("max_depth", 0.0) or 0.0
            flood_vol = stats.get("flooding_volume", 0.0) or 0.0
            flood_hours = stats.get("time_flooded", 0.0) or 0.0
            flood_minutes = flood_hours * 60.0

            raw_peak = stats.get("time_max_depth")
            if raw_peak and sim_start:
                peak_dt = sim_start + timedelta(days=raw_peak - int(sim_start.toordinal()))
                time_to_peak = max(0.0, (peak_dt - sim_start).total_seconds() / 60.0)
            else:
                time_to_peak = None

            depth_ratio = (max_depth / full_depth) if (full_depth and full_depth > 0) else None
            flooded = flood_vol > 0
            if flooded:
                total_flooded += 1

            node_records.append({
                "result_id": new_id(),
                "delta_inflow_lps": safe_round(scenario_reference_inflow_lps(node_id), 6),
                "inflow_multiplier": safe_round(hydrograph_multiplier if hydrograph is not None else inflow_multiplier, 6),
                "node_id": node_id,
                "flooded": int(flooded),
                "flooding_volume_m3": safe_round(flood_vol),
                "flooding_duration_min": safe_round(flood_minutes, 2),
                "max_depth_m": safe_round(max_depth),
                "max_depth_ratio": safe_round(depth_ratio),
                "time_to_peak_min": safe_round(time_to_peak, 2),
                "depth_rate_m_per_min": safe_round(depth_rate_tracker[node_id]["max_rate"]),
            })

            if target_node_set is None or node_id in target_node_set:
                run_inputs.append({
                    "input_id": new_id(),
                    "delta_inflow_lps": safe_round(scenario_reference_inflow_lps(node_id), 6),
                    "inflow_multiplier": safe_round(hydrograph_multiplier if hydrograph is not None else inflow_multiplier, 6),
                    "node_uid": node_id,
                })

        link_records = []
        for link in links:
            link_id = link.linkid
            conduit_stats = {}
            try:
                conduit_stats = link.conduit_statistics
            except AttributeError:
                pass

            peak_flow_lps = conduit_stats.get("peak_flow", 0.0) or 0.0
            peak_vel_mps = conduit_stats.get("peak_velocity", 0.0) or 0.0
            peak_depth_m = conduit_stats.get("peak_depth", 0.0) or 0.0
            time_full_hours = conduit_stats.get("time_full_flow", 0.0) or 0.0
            surcharged = time_full_hours > 0

            capacity = link_static.get(link_id, {}).get("full_flow_capacity_lps")
            cap_ratio = round(peak_flow_lps / capacity, 4) if (capacity and capacity > 0) else None

            link_records.append({
                "result_id": new_id(),
                "delta_inflow_lps": safe_round(hydrograph_multiplier if hydrograph is not None else inflow_multiplier, 6),
                "inflow_multiplier": safe_round(hydrograph_multiplier if hydrograph is not None else inflow_multiplier, 6),
                "link_id": link_id,
                "max_flow_lps": safe_round(peak_flow_lps, 4),
                "max_velocity_mps": safe_round(peak_vel_mps),
                "max_depth_m": safe_round(peak_depth_m),
                "max_capacity_ratio": cap_ratio,
                "surcharged": int(surcharged),
                "time_full_flow_hrs": safe_round(time_full_hours, 4),
            })

    total_nodes = len(node_records)
    total_flood_vol = sum(record["flooding_volume_m3"] or 0 for record in node_records)
    pct_flooded = (total_flooded / total_nodes * 100) if total_nodes > 0 else 0.0
    time_to_first = (
        round(first_flood_step * timestep_sec / 60.0, 2)
        if first_flood_step and timestep_sec
        else None
    )
    resilience = round(1.0 - (total_flooded / total_nodes), 4) if total_nodes > 0 else 1.0

    summary = {
        "summary_id": new_id(),
        "delta_inflow_lps": safe_round(hydrograph_multiplier if hydrograph is not None else inflow_multiplier, 6),
        "inflow_multiplier": safe_round(hydrograph_multiplier if hydrograph is not None else inflow_multiplier, 6),
        "total_nodes": total_nodes,
        "total_flooded_nodes": total_flooded,
        "total_flooding_volume_m3": round(total_flood_vol, 4),
        "pct_flooded_nodes": round(pct_flooded, 2),
        "time_to_first_flood_min": time_to_first,
        "resilience_index": resilience,
    }

    return {
        "node_records": node_records,
        "link_records": link_records,
        "run_inputs": run_inputs,
        "summary": summary,
    }
