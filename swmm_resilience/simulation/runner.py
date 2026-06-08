"""
Simulation-only logic: PySWMM setup, topology extraction and hydraulic results.
"""

import json
import tempfile
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from statistics import median

from ..config import (
    INPUT_VALIDATION_MAX_REASONABLE_JUNCTION_DEPTH_M,
    INPUT_VALIDATION_MAX_SUSPICIOUS_JUNCTION_DEPTH_FRACTION,
    INPUT_VALIDATION_MIN_JUNCTIONS,
    STRICT_INPUT_VALIDATION,
    USE_SWMM_API_RPT_RESULTS,
)
from ..utils import (
    circular_full_flow_lps,
    new_id,
    node_type_str,
    safe_round,
)
from .swmm_api_io import (
    get_base_node_inflows_lps,
    get_node_inflow_profiles,
    load_inp,
    read_node_flooding_summary,
    write_scaled_inp,
)


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


def _remove_swmm_temp_files(inp_path: Path):
    for suffix in (".inp", ".rpt", ".out"):
        with suppress(OSError):
            inp_path.with_suffix(suffix).unlink()


def _agg_link_field(link_static: dict, link_ids: list, field: str) -> list:
    return [
        link_static[link_id][field]
        for link_id in link_ids
        if link_static.get(link_id, {}).get(field) is not None
    ]


def _safe_avg(values):
    return round(sum(values) / len(values), 6) if values else None


def _safe_max(values):
    return round(max(values), 6) if values else None


def _safe_min(values):
    return round(min(values), 6) if values else None


def _safe_sum(values):
    return round(sum(values), 6) if values else None


def _flood_volume_from_timeseries_m3(node_timeseries_records: list[dict]) -> dict[str, float]:
    """Integrate node flooding time series from L/s to m3."""
    rows_by_node = defaultdict(list)
    for record in node_timeseries_records:
        node_id = record.get("node_id")
        if node_id is not None:
            rows_by_node[str(node_id)].append(record)

    volumes = {}
    for node_id, records in rows_by_node.items():
        previous_time_sec = 0.0
        total_litres = 0.0
        for record in sorted(records, key=lambda row: row.get("time_sec") or 0.0):
            try:
                time_sec = float(record.get("time_sec") or 0.0)
                flooding_lps = float(record.get("flooding_lps") or 0.0)
            except (TypeError, ValueError):
                continue
            dt_sec = time_sec - previous_time_sec
            if dt_sec > 0:
                total_litres += flooding_lps * dt_sec
            previous_time_sec = time_sec
        volumes[node_id] = round(total_litres / 1000.0, 6)
    return volumes


def _merge_rpt_flooding_metrics(node_records: list[dict], rpt_df) -> None:
    """Overlay node flooding metrics parsed from the SWMM .rpt summary."""
    if rpt_df is None:
        return

    def _is_positive(value) -> bool:
        try:
            metric = float(value)
        except (TypeError, ValueError):
            return False
        return metric == metric and metric > 0

    rpt_lookup = {str(row["node_id"]): row for _, row in rpt_df.iterrows()}
    for record in node_records:
        rpt_row = rpt_lookup.get(str(record["node_id"]))
        if rpt_row is not None:
            volume = rpt_row.get("flooding_volume_m3")
            try:
                fvol = float(volume)
                if fvol == fvol:
                    record["total_flood_volume_m3"] = safe_round(fvol, 6)
            except (TypeError, ValueError):
                pass

            duration = rpt_row.get("flooding_duration_min")
            try:
                fdur = float(duration)
                if fdur == fdur:
                    record["flooding_duration_min"] = safe_round(fdur, 2)
            except (TypeError, ValueError):
                pass

        record["flooded"] = int(
            _is_positive(record.get("peak_flooding_lps"))
            or _is_positive(record.get("total_flood_volume_m3"))
            or _is_positive(record.get("flooding_duration_min"))
        )


def _validate_network_geometry(inp_file: str, junction_depths: list[tuple[str, float]]):
    """Fail fast when a network's junction depths are implausibly large."""
    if not STRICT_INPUT_VALIDATION:
        return
    if len(junction_depths) < INPUT_VALIDATION_MIN_JUNCTIONS:
        return

    too_deep = [
        (node_id, depth)
        for node_id, depth in junction_depths
        if depth > INPUT_VALIDATION_MAX_REASONABLE_JUNCTION_DEPTH_M
    ]
    deep_fraction = len(too_deep) / len(junction_depths)
    if deep_fraction < INPUT_VALIDATION_MAX_SUSPICIOUS_JUNCTION_DEPTH_FRACTION:
        return

    median_depth = median(depth for _, depth in junction_depths)
    examples = ", ".join(
        f"{node_id}={depth:.3f} m"
        for node_id, depth in too_deep[:5]
    )
    raise ValueError(
        "La red "
        f"'{Path(inp_file).name}' tiene profundidades utiles de junction "
        f"sospechosamente altas: mediana={median_depth:.3f} m y "
        f"{len(too_deep)}/{len(junction_depths)} nodos superan "
        f"{INPUT_VALIDATION_MAX_REASONABLE_JUNCTION_DEPTH_M:.1f} m. "
        "Con esos MaxDepth, SWMM puede no reportar flooding aunque el hidrograma "
        "si se haya multiplicado correctamente. Esto vuelve no comparables casos "
        "como Qx1*3 frente a Qx3 si las geometrías no coinciden. "
        f"Ejemplos: {examples}. "
        "Revisa la seccion [JUNCTIONS] del .inp y corrige MaxDepth antes de correr. "
        "Si quieres omitir este bloqueo conscientemente, desactiva "
        "STRICT_INPUT_VALIDATION en swmm_resilience/config.py."
    )


def extract_static_topology(inp_file: str, net_hash: str, Nodes, Links, Simulation):
    """Extract static network topology and aggregated node/link features from SWMM."""
    node_records = []
    link_records = []
    link_static = {}
    junction_depths = []

    inp = load_inp(inp_file)
    base_node_inflows_lps = get_base_node_inflows_lps(inp)
    node_inflow_profiles = get_node_inflow_profiles(inp)

    conduit_rows = {}
    if "CONDUITS" in inp:
        for link_id, c in inp["CONDUITS"].items():
            conduit_rows[str(link_id)] = {
                "inlet_node": c.from_node,
                "outlet_node": c.to_node,
                "length_m": float(c.length) if c.length is not None else None,
                "roughness": float(c.roughness) if c.roughness is not None else None,
                "inlet_offset": float(c.offset_upstream) if c.offset_upstream is not None else 0.0,
                "outlet_offset": float(c.offset_downstream) if c.offset_downstream is not None else 0.0,
            }
    xsection_rows = {}
    if "XSECTIONS" in inp:
        for link_id, x in inp["XSECTIONS"].items():
            xsection_rows[str(link_id)] = {
                "shape": x.shape,
                "geom1": float(x.height) if x.height is not None else None,
            }

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
            full_depth = node.full_depth
            if node_type == "junction" and full_depth is not None:
                junction_depths.append((node_id, float(full_depth)))

            upstream_ids = upstream_links.get(node_id, [])
            downstream_ids = downstream_links.get(node_id, [])

            upstream_diams = _agg_link_field(link_static, upstream_ids, "diameter_m")
            upstream_slopes = [abs(value) for value in _agg_link_field(link_static, upstream_ids, "slope_m_per_m") if value is not None]
            upstream_caps = _agg_link_field(link_static, upstream_ids, "full_flow_capacity_lps")

            downstream_diams = _agg_link_field(link_static, downstream_ids, "diameter_m")
            downstream_slopes = [abs(value) for value in _agg_link_field(link_static, downstream_ids, "slope_m_per_m") if value is not None]
            downstream_caps = _agg_link_field(link_static, downstream_ids, "full_flow_capacity_lps")

            node_records.append({
                "node_uid": node_id,
                "network_hash": net_hash,
                "invert_elev_m": safe_round(node.invert_elevation),
                "full_depth_m": safe_round(full_depth),
                "base_inflow_lps": safe_round(base_node_inflows_lps.get(node_id, 0.0), 6),
                "node_type": node_type,
                "in_degree": len(upstream_ids),
                "out_degree": len(downstream_ids),
                "upstream_pipes_count": len(upstream_ids),
                "upstream_diam_max_m": _safe_max(upstream_diams),
                "upstream_diam_min_m": _safe_min(upstream_diams),
                "upstream_diam_avg_m": _safe_avg(upstream_diams),
                "upstream_slope_avg": _safe_avg(upstream_slopes),
                "upstream_slope_max": _safe_max(upstream_slopes),
                "upstream_capacity_lps": _safe_sum(upstream_caps),
                "downstream_pipes_count": len(downstream_ids),
                "downstream_diam_max_m": _safe_max(downstream_diams),
                "downstream_diam_min_m": _safe_min(downstream_diams),
                "downstream_diam_avg_m": _safe_avg(downstream_diams),
                "downstream_slope_avg": _safe_avg(downstream_slopes),
                "downstream_slope_max": _safe_max(downstream_slopes),
                "downstream_capacity_lps": _safe_sum(downstream_caps),
            })

    _validate_network_geometry(inp_file, junction_depths)

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
    scenario_mode=None,
    run_id=None,
    network_hash=None,
):
    """Execute one SWMM run scaling embedded .inp lateral inflows by a multiplier."""

    first_flood_elapsed_min = None
    step_count = 0
    timestep_sec = None
    depth_rate_tracker = {}
    peak_flooding_lps_tracker = {}
    node_timeseries_records = []
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
        _tmp_dir = Path(tempfile.mkdtemp(prefix="swmm_scaled_"))
        _tmp_inp = _tmp_dir / f"swmm_scaled_{Path(inp_file).stem}.inp"
        scaled_inp_path = write_scaled_inp(
            inp_file,
            inflow_multiplier,
            target_node_set,
            _tmp_inp,
            scenario_mode=scenario_mode,
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
            depth_rate_tracker[node.nodeid] = {"prev": None, "max_rate": 0.0}
            peak_flooding_lps_tracker[node.nodeid] = 0.0
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
            elapsed_sec = (sim.current_time - sim_start).total_seconds()
            if timestep_sec is None:
                timestep_sec = elapsed_sec if elapsed_sec > 0 else 1.0
            elapsed_min = elapsed_sec / 60.0

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

                if first_flood_elapsed_min is None and node.flooding > 0:
                    first_flood_elapsed_min = elapsed_min

                depth = node.depth
                tracker = depth_rate_tracker[node_id]
                if tracker["prev"] is not None and timestep_sec and timestep_sec > 0:
                    rate = (depth - tracker["prev"]) / (timestep_sec / 60.0)
                    if rate > tracker["max_rate"]:
                        tracker["max_rate"] = rate
                tracker["prev"] = depth

                total_outflow = node.total_outflow or 0.0
                if total_outflow > outflow_tracker[node_id]["max_total_outflow_lps"]:
                    outflow_tracker[node_id]["max_total_outflow_lps"] = total_outflow
                    outflow_tracker[node_id]["time_to_peak_outflow_min"] = elapsed_min

                full_depth = node.full_depth
                depth_ratio = (
                    depth / full_depth
                    if (full_depth and full_depth > 0)
                    else None
                )
                flooding_lps = node.flooding or 0.0
                if flooding_lps > peak_flooding_lps_tracker[node_id]:
                    peak_flooding_lps_tracker[node_id] = flooding_lps
                node_timeseries_records.append({
                    "run_id": run_id,
                    "network_hash": network_hash,
                    "node_id": node_id,
                    "step_index": step_count,
                    "time_sec": safe_round(elapsed_sec, 4),
                    "time_min": safe_round(elapsed_min, 6),
                    "total_inflow_lps": safe_round(node.total_inflow or 0.0, 6),
                    "lateral_inflow_lps": safe_round(node.lateral_inflow or 0.0, 6),
                    "depth_m": safe_round(depth, 6),
                    "depth_ratio": safe_round(depth_ratio, 6),
                    "flooding_lps": safe_round(flooding_lps, 6),
                    "total_outflow_lps": safe_round(total_outflow, 6),
                    "failed_now": int(flooding_lps > 0),
                })

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

        fallback_flood_volume_m3 = _flood_volume_from_timeseries_m3(node_timeseries_records)
        node_records = []
        run_inputs = []
        total_flooded = 0

        for node in nodes:
            node_id = node.nodeid
            stats = node.statistics

            full_depth = node.full_depth
            max_depth = stats.get("max_depth", 0.0) or 0.0
            flood_hours = stats.get("time_flooded", 0.0) or 0.0
            flood_minutes = flood_hours * 60.0
            peak_flooding_lps = peak_flooding_lps_tracker[node_id]
            total_flood_volume_m3 = fallback_flood_volume_m3.get(node_id, 0.0)

            raw_peak = stats.get("time_max_depth")
            if raw_peak is not None and sim_start:
                sim_start_ord = (
                    sim_start.toordinal()
                    + (sim_start.hour * 3600 + sim_start.minute * 60 + sim_start.second)
                    / 86400.0
                )
                time_to_peak = max(0.0, (raw_peak - sim_start_ord) * 24.0 * 60.0)
            else:
                time_to_peak = None

            depth_ratio = (max_depth / full_depth) if (full_depth and full_depth > 0) else None
            flooded = peak_flooding_lps > 0 or total_flood_volume_m3 > 0
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
                "peak_flooding_lps": safe_round(peak_flooding_lps, 4),
                "total_flood_volume_m3": safe_round(total_flood_volume_m3, 6),
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
                "delta_inflow_lps": None,
                "inflow_multiplier": safe_round(inflow_multiplier, 6),
                "link_id": link_id,
                "max_flow_lps": safe_round(peak_flow_lps, 4),
                "max_velocity_mps": safe_round(peak_vel_mps),
                "max_depth_m": safe_round(peak_depth_m),
                "max_capacity_ratio": cap_ratio,
                "surcharged": int(surcharged),
                "time_full_flow_hrs": safe_round(time_full_hours, 4),
            })
    if USE_SWMM_API_RPT_RESULTS:
        _rpt_path = Path(simulation_inp).with_suffix(".rpt")
        _merge_rpt_flooding_metrics(node_records, read_node_flooding_summary(_rpt_path))
        total_flooded = sum(r["flooded"] for r in node_records)

    if scaled_inp_path is not None:
        _remove_swmm_temp_files(scaled_inp_path)
        with suppress(OSError):
            scaled_inp_path.parent.rmdir()

    total_nodes = len(node_records)
    total_peak_flooding_lps = sum(record["peak_flooding_lps"] or 0 for record in node_records)
    total_flood_volume_m3 = sum(record["total_flood_volume_m3"] or 0 for record in node_records)
    pct_flooded = (total_flooded / total_nodes * 100) if total_nodes > 0 else 0.0
    time_to_first = (
        round(first_flood_elapsed_min, 2)
        if first_flood_elapsed_min is not None
        else None
    )
    resilience = round(1.0 - (total_flooded / total_nodes), 4) if total_nodes > 0 else 1.0

    summary = {
        "summary_id": new_id(),
        "inflow_multiplier": safe_round(inflow_multiplier, 6),
        "total_nodes": total_nodes,
        "failed_nodes_count": total_flooded,
        "total_peak_flooding_lps": round(total_peak_flooding_lps, 4),
        "total_flood_volume_m3": round(total_flood_volume_m3, 6),
        "pct_flooded_nodes": round(pct_flooded, 2),
        "time_to_first_flood_min": time_to_first,
        "resilience_index": resilience,
    }

    return {
        "node_records": node_records,
        "link_records": link_records,
        "run_inputs": run_inputs,
        "node_timeseries_records": node_timeseries_records,
        "summary": summary,
    }


def run_simulation_simple(inp_path: Path, factor: float, run_dir: Path) -> Path:
    """Simple single-factor SWMM run. Scales inflows, runs, returns .rpt path.

    Used by batch.py (origin/main interface). The original .inp is never modified.
    """
    from pyswmm import Simulation as _Simulation

    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_inp = run_dir / f"factor_{factor:.4f}.inp"
    write_scaled_inp(str(inp_path), factor, None, str(tmp_inp), scenario_mode="timeseries")

    with _Simulation(str(tmp_inp)) as sim:
        for _ in sim:
            pass

    rpt_path = tmp_inp.with_suffix(".rpt")
    with suppress(OSError):
        tmp_inp.unlink()

    if not rpt_path.exists():
        raise FileNotFoundError(f"SWMM no genero el archivo .rpt esperado: {rpt_path}")
    return rpt_path
