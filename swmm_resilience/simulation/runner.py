"""
Simulation-only logic: PySWMM setup, topology extraction and hydraulic results.
"""

import json
import tempfile
from collections import defaultdict
from contextlib import suppress
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
    inflow_defs = {}
    timeseries_rows = defaultdict(list)
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
            node_id = parts[0]
            try:
                baseline = float(parts[6]) if len(parts) > 6 else 0.0
                inflow_rows[node_id] += baseline
            except ValueError:
                baseline = 0.0
            if len(parts) >= 3 and parts[1].upper() == "FLOW":
                try:
                    mfactor = float(parts[4]) if len(parts) > 4 else 1.0
                except ValueError:
                    mfactor = 1.0
                inflow_defs[node_id] = {
                    "timeseries": parts[2],
                    "mfactor": mfactor,
                    "baseline": baseline,
                }
        elif current_section == "[TIMESERIES]" and len(parts) >= 3:
            series_id = parts[0]
            if ":" in parts[1]:
                time_token = parts[1]
                value_token = parts[2]
            elif len(parts) >= 4:
                time_token = parts[2]
                value_token = parts[3]
            else:
                continue
            try:
                timeseries_rows[series_id].append((
                    _time_token_to_minutes(time_token),
                    float(value_token),
                ))
            except ValueError:
                continue

    node_inflow_profiles = {}
    for node_id, inflow_def in inflow_defs.items():
        points = sorted(timeseries_rows.get(inflow_def["timeseries"], []))
        node_inflow_profiles[node_id] = {
            "timeseries": inflow_def["timeseries"],
            "points": points,
            "mfactor": inflow_def["mfactor"],
            "baseline": inflow_def["baseline"],
        }

    return conduit_rows, xsection_rows, dict(inflow_rows), node_inflow_profiles


def _time_token_to_minutes(token: str) -> float:
    parts = token.split(":")
    if len(parts) == 2:
        hours, minutes = parts
        seconds = 0.0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Formato de hora no reconocido: {token}")
    return float(hours) * 60.0 + float(minutes) + float(seconds) / 60.0


def _profile_value_lps(profile, elapsed_min: float) -> float:
    if not profile:
        return 0.0
    points = profile.get("points", [])
    baseline = profile.get("baseline", 0.0) or 0.0
    mfactor = profile.get("mfactor", 1.0) or 1.0
    if not points:
        return baseline
    if elapsed_min <= points[0][0]:
        return baseline + points[0][1] * mfactor

    for left, right in zip(points, points[1:]):
        left_min, left_flow = left
        right_min, right_flow = right
        if elapsed_min <= right_min:
            span = right_min - left_min
            if span <= 0:
                return baseline + right_flow * mfactor
            fraction = (elapsed_min - left_min) / span
            return baseline + (left_flow + fraction * (right_flow - left_flow)) * mfactor

    return baseline + points[-1][1] * mfactor


def _write_scaled_inp(inp_file: str, inflow_multiplier: float, target_node_set, node_inflow_profiles) -> Path:
    target_series = {
        profile["timeseries"]
        for node_id, profile in node_inflow_profiles.items()
        if target_node_set is None or node_id in target_node_set
    }
    temp_dir = Path("C:/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        errors="replace",
        suffix=".inp",
        prefix="swmm_scaled_",
        dir=temp_dir,
        delete=False,
    )
    temp_path = Path(handle.name)
    current_section = None

    with handle:
        for raw_line in Path(inp_file).read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                current_section = line.upper()
                handle.write(raw_line + "\n")
                continue
            if current_section != "[TIMESERIES]" or not line or line.startswith(";;"):
                handle.write(raw_line + "\n")
                continue

            parts = line.split()
            if len(parts) < 3 or parts[0] not in target_series:
                handle.write(raw_line + "\n")
                continue

            value_index = 2 if ":" in parts[1] else 3
            if value_index >= len(parts):
                handle.write(raw_line + "\n")
                continue
            try:
                parts[value_index] = f"{float(parts[value_index]) * inflow_multiplier:.6f}"
            except ValueError:
                handle.write(raw_line + "\n")
                continue
            handle.write(" ".join(parts) + "\n")

    return temp_path


def _remove_swmm_temp_files(inp_path: Path):
    for suffix in (".inp", ".rpt", ".out"):
        with suppress(OSError):
            inp_path.with_suffix(suffix).unlink()


def extract_static_topology(inp_file: str, net_hash: str, Nodes, Links, Simulation):
    """Extract static network topology and aggregated node/link features from SWMM."""
    node_records = []
    link_records = []
    link_static = {}
    conduit_rows, xsection_rows, base_node_inflows_lps, node_inflow_profiles = _parse_inp_sections(inp_file)

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
        "node_inflow_profiles": node_inflow_profiles,
    }


def run_simulation(
    inp_file,
    inflow_multiplier,
    link_static,
    Nodes,
    Links,
    Simulation,
    target_nodes=None,
    node_inflow_profiles=None,
):
    """Execute one SWMM run scaling embedded .inp lateral inflows by a multiplier."""

    first_flood_step = None
    step_count = 0
    timestep_sec = None
    depth_rate_tracker = {}
    node_inflow_profiles = node_inflow_profiles or {}
    if target_nodes is None:
        target_node_set = None
    elif isinstance(target_nodes, str):
        target_node_set = {target_nodes}
    else:
        target_node_set = {str(node_id) for node_id in target_nodes}

    scaled_inp_path = None
    simulation_inp = inp_file
    if inflow_multiplier != 1.0 or target_node_set is not None:
        scaled_inp_path = _write_scaled_inp(
            inp_file,
            inflow_multiplier,
            target_node_set,
            node_inflow_profiles,
        )
        simulation_inp = str(scaled_inp_path)

    with Simulation(simulation_inp) as sim:
        sim_start = sim.start_time
        nodes = list(Nodes(sim))
        links = list(Links(sim))
        link_inlet_nodes = {}
        outflow_tracker = {}
        input_tracker = {}
        downstream_link_peak_flows = defaultdict(dict)

        for node in nodes:
            depth_rate_tracker[node.nodeid] = {"prev": 0.0, "max_rate": 0.0}
            outflow_tracker[node.nodeid] = {
                "max_total_outflow_lps": 0.0,
                "time_to_peak_outflow_min": None,
            }
            input_tracker[node.nodeid] = {
                "peak_embedded_lateral_lps": 0.0,
                "peak_added_lps": 0.0,
                "peak_scaled_lateral_lps": 0.0,
            }

        for link in links:
            link_inlet_nodes[link.linkid] = getattr(link, "inlet_node", None)

        for _ in sim:
            step_count += 1
            if timestep_sec is None:
                dt = (sim.current_time - sim_start).total_seconds()
                timestep_sec = dt if dt > 0 else 1.0
            elapsed_min = (sim.current_time - sim_start).total_seconds() / 60.0

            for node in nodes:
                node_id = node.nodeid
                applies_to_node = target_node_set is None or node_id in target_node_set
                embedded_lateral_lps = _profile_value_lps(
                    node_inflow_profiles.get(node_id),
                    elapsed_min,
                )
                applied_lps = (
                    embedded_lateral_lps * (inflow_multiplier - 1.0)
                    if applies_to_node
                    else 0.0
                )
                scaled_lateral_lps = embedded_lateral_lps + applied_lps

                input_metrics = input_tracker[node_id]
                if abs(embedded_lateral_lps) > abs(input_metrics["peak_embedded_lateral_lps"]):
                    input_metrics["peak_embedded_lateral_lps"] = embedded_lateral_lps
                if abs(applied_lps) > abs(input_metrics["peak_added_lps"]):
                    input_metrics["peak_added_lps"] = applied_lps
                if abs(scaled_lateral_lps) > abs(input_metrics["peak_scaled_lateral_lps"]):
                    input_metrics["peak_scaled_lateral_lps"] = scaled_lateral_lps

                if first_flood_step is None and node.flooding > 0:
                    first_flood_step = step_count

                depth = node.depth
                tracker = depth_rate_tracker[node_id]
                if timestep_sec > 0:
                    rate = (depth - tracker["prev"]) / (timestep_sec / 60.0)
                    if rate > tracker["max_rate"]:
                        tracker["max_rate"] = rate
                tracker["prev"] = depth

                total_outflow = node.total_outflow or 0.0
                if total_outflow > outflow_tracker[node_id]["max_total_outflow_lps"]:
                    outflow_tracker[node_id]["max_total_outflow_lps"] = total_outflow
                    outflow_tracker[node_id]["time_to_peak_outflow_min"] = elapsed_min

            for link in links:
                inlet_node = link_inlet_nodes.get(link.linkid)
                if not inlet_node:
                    continue
                flow_lps = link.flow or 0.0
                if flow_lps <= 0:
                    continue
                current_peak = downstream_link_peak_flows[inlet_node].get(link.linkid, 0.0)
                if flow_lps > current_peak:
                    downstream_link_peak_flows[inlet_node][link.linkid] = flow_lps

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

            outflow_metrics = outflow_tracker[node_id]
            input_metrics = input_tracker[node_id]
            downstream_flow_map = {
                link_id: safe_round(flow_lps, 4)
                for link_id, flow_lps in sorted(downstream_link_peak_flows.get(node_id, {}).items())
            }

            node_records.append({
                "result_id": new_id(),
                "delta_inflow_lps": safe_round(input_metrics["peak_added_lps"], 6),
                "inflow_multiplier": safe_round(inflow_multiplier, 6),
                "node_id": node_id,
                "flooded": int(flooded),
                "flooding_volume_m3": safe_round(flood_vol),
                "flooding_duration_min": safe_round(flood_minutes, 2),
                "max_depth_m": safe_round(max_depth),
                "max_depth_ratio": safe_round(depth_ratio),
                "time_to_peak_min": safe_round(time_to_peak, 2),
                "depth_rate_m_per_min": safe_round(depth_rate_tracker[node_id]["max_rate"]),
                "max_total_outflow_lps": safe_round(outflow_metrics["max_total_outflow_lps"], 4),
                "time_to_peak_outflow_min": safe_round(outflow_metrics["time_to_peak_outflow_min"], 2),
                "downstream_link_peak_flows_lps_json": json.dumps(
                    downstream_flow_map,
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            })

            if target_node_set is None or node_id in target_node_set:
                run_inputs.append({
                    "input_id": new_id(),
                    "delta_inflow_lps": safe_round(input_metrics["peak_added_lps"], 6),
                    "inflow_multiplier": safe_round(inflow_multiplier, 6),
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
                "delta_inflow_lps": safe_round(inflow_multiplier, 6),
                "inflow_multiplier": safe_round(inflow_multiplier, 6),
                "link_id": link_id,
                "max_flow_lps": safe_round(peak_flow_lps, 4),
                "max_velocity_mps": safe_round(peak_vel_mps),
                "max_depth_m": safe_round(peak_depth_m),
                "max_capacity_ratio": cap_ratio,
                "surcharged": int(surcharged),
                "time_full_flow_hrs": safe_round(time_full_hours, 4),
            })
    if scaled_inp_path is not None:
        _remove_swmm_temp_files(scaled_inp_path)

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
        "inflow_multiplier": safe_round(inflow_multiplier, 6),
        "total_nodes": total_nodes,
        "failed_nodes_count": total_flooded,
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
