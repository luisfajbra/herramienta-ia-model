import pandas as pd
from pathlib import Path

from ..simulation.swmm_api_io import load_inp


def _get_peak_inflows(inp) -> dict:
    """Return {node_id: peak_flow_lps} — max value from the node's timeseries (factor=1)."""
    if "INFLOWS" not in inp:
        return {}
    ts_map = dict(inp["TIMESERIES"]) if "TIMESERIES" in inp else {}
    result = {}
    for (node, constituent), inflow in inp["INFLOWS"].items():
        if str(constituent).upper() != "FLOW":
            continue
        node = str(node)
        ts_name = inflow.time_series
        ts_name_str = str(ts_name).strip() if ts_name is not None else ""
        if ts_name_str and ts_name_str not in ('""', "''") and ts_name in ts_map:
            values = [v for _, v in ts_map[ts_name].data]
            result[node] = max(values) if values else 0.0
        elif inflow.base_value:
            result[node] = float(inflow.base_value)
        else:
            result[node] = 0.0
    return result


def extract_static_features(inp_path: Path) -> pd.DataFrame:
    """Extract per-junction static features from a SWMM .inp file.

    Returns 1 row per junction (outfalls excluded). Columns:
        node_id, elev_fondo, prof_max, n_tuberias_in,
        diam_max_in, diam_max_out, pendiente_max_in, pendiente_out,
        base_inflow_lps, coord_x, coord_y

    NaN for diam_max_in and pendiente_max_in on headwater nodes (no upstream pipes).
    These NaNs are propagated to the dataset and imputed by the ML pipeline.
    """
    inp = load_inp(inp_path)

    junctions = {}
    if "JUNCTIONS" in inp:
        for nid, j in inp["JUNCTIONS"].items():
            junctions[str(nid)] = {
                "elev_fondo": float(j.elevation),
                "prof_max": float(j.depth_max),
            }

    coords = {}
    if "COORDINATES" in inp:
        for nid, c in inp["COORDINATES"].items():
            coords[str(nid)] = (float(c.x), float(c.y))

    xsections = {}
    if "XSECTIONS" in inp:
        for lid, x in inp["XSECTIONS"].items():
            xsections[str(lid)] = float(x.height) if x.height is not None else None

    conduits = []
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            conduits.append({
                "link_id": str(lid),
                "from_node": str(c.from_node),
                "to_node": str(c.to_node),
                "length": float(c.length) if c.length else 1.0,
                "inlet_offset": float(c.offset_upstream) if c.offset_upstream else 0.0,
                "outlet_offset": float(c.offset_downstream) if c.offset_downstream else 0.0,
            })

    peak_inflows = _get_peak_inflows(inp)

    in_conduits = {nid: [] for nid in junctions}
    out_conduits = {nid: [] for nid in junctions}
    for c in conduits:
        if c["to_node"] in in_conduits:
            in_conduits[c["to_node"]].append(c)
        if c["from_node"] in out_conduits:
            out_conduits[c["from_node"]].append(c)

    def compute_slope(c):
        fe = junctions.get(c["from_node"], {}).get("elev_fondo")
        te = junctions.get(c["to_node"], {}).get("elev_fondo")
        if fe is None or te is None or c["length"] <= 0:
            return None
        return (fe + c["inlet_offset"] - te - c["outlet_offset"]) / c["length"]

    rows = []
    for nid, nd in junctions.items():
        ins = in_conduits[nid]
        outs = out_conduits[nid]

        d_in = [xsections[c["link_id"]] for c in ins if xsections.get(c["link_id"]) is not None]
        d_out = [xsections[c["link_id"]] for c in outs if xsections.get(c["link_id"]) is not None]
        s_in = [s for c in ins if (s := compute_slope(c)) is not None]
        s_out = [s for c in outs if (s := compute_slope(c)) is not None]

        cx, cy = coords.get(nid, (None, None))
        rows.append({
            "node_id": nid,
            "elev_fondo": nd["elev_fondo"],
            "prof_max": nd["prof_max"],
            "diam_max_in": max(d_in) if d_in else None,
            "diam_max_out": max(d_out) if d_out else None,
            "pendiente_max_in": max(s_in) if s_in else None,
            "pendiente_out": s_out[0] if s_out else None,
            "base_inflow_lps": peak_inflows.get(nid, 0.0),
            "coord_x": cx,
            "coord_y": cy,
        })

    return pd.DataFrame(rows)
