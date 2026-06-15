import networkx as nx
import pandas as pd
from pathlib import Path

from ..simulation.swmm_api_io import load_inp
from ..utils import circular_full_flow_lps


def build_network_graph(inp):
    """Public alias of _build_graph for use outside this module."""
    return _build_graph(inp)


def _build_graph(inp):
    """Build directed graph (from_node → to_node) and return (G, outfalls_set)."""
    G = nx.DiGraph()
    outfalls = set()
    if "JUNCTIONS" in inp:
        for nid in inp["JUNCTIONS"]:
            G.add_node(str(nid))
    if "OUTFALLS" in inp:
        for nid in inp["OUTFALLS"]:
            G.add_node(str(nid))
            outfalls.add(str(nid))
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            G.add_edge(
                str(c.from_node), str(c.to_node),
                length=float(c.length) if c.length else 0.0,
                link_id=str(lid),
            )
    return G, outfalls


def compute_topology_features(static_df: pd.DataFrame, inp_path: Path) -> pd.DataFrame:
    """Add topology columns to static_df and return the augmented DataFrame.

    Adds: dist_outfall_m, n_nodos_aguas_arriba, q_pico_acum_base, upstream_capacity_lps

    dist_outfall_m: NaN if no path to outfall exists (disconnected node).
    upstream_capacity_lps: Manning full-flow capacity of immediate upstream conduits.
    """
    inp = load_inp(inp_path)
    G, outfalls = _build_graph(inp)

    xsections = {}
    if "XSECTIONS" in inp:
        for lid, x in inp["XSECTIONS"].items():
            xsections[str(lid)] = float(x.height) if x.height is not None else None

    conduit_meta = {}
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            conduit_meta[str(lid)] = {
                "from_node": str(c.from_node),
                "to_node": str(c.to_node),
                "length": float(c.length) if c.length else 1.0,
                "roughness": float(c.roughness) if c.roughness else None,
                "inlet_offset": float(c.offset_upstream) if c.offset_upstream else 0.0,
                "outlet_offset": float(c.offset_downstream) if c.offset_downstream else 0.0,
            }

    junction_elev = dict(zip(static_df["node_id"], static_df["elev_fondo"]))
    base_inflows = dict(zip(static_df["node_id"], static_df["base_inflow_lps"]))

    rows = []
    for _, row in static_df.iterrows():
        nid = row["node_id"]

        # dist_outfall_m: shortest weighted path to any outfall
        dist = None
        for outfall in outfalls:
            if nid == outfall:
                dist = 0.0
                break
            if G.has_node(nid) and G.has_node(outfall):
                try:
                    d = nx.shortest_path_length(G, nid, outfall, weight="length")
                    dist = d if dist is None else min(dist, d)
                except nx.NetworkXNoPath:
                    pass

        # n_nodos_aguas_arriba: all nodes that drain into this one
        ancestors = nx.ancestors(G, nid) if G.has_node(nid) else set()
        n_upstream = len(ancestors)

        # q_pico_acum_base: own inflow + all upstream inflows
        q_acum = sum(base_inflows.get(n, 0.0) for n in ancestors | {nid})

        # upstream_capacity_lps: Manning full-flow capacity of immediate upstream conduits
        cap_total = 0.0
        for pred in G.predecessors(nid):
            lid = G[pred][nid].get("link_id")
            if lid is None:
                continue
            diam = xsections.get(lid)
            meta = conduit_meta.get(lid, {})
            roughness = meta.get("roughness")
            fn_elev = junction_elev.get(meta.get("from_node"))
            tn_elev = junction_elev.get(meta.get("to_node"))
            length = meta.get("length", 1.0)
            if fn_elev is not None and tn_elev is not None and length > 0:
                slope = abs(
                    (fn_elev + meta.get("inlet_offset", 0.0)
                     - tn_elev - meta.get("outlet_offset", 0.0)) / length
                )
            else:
                slope = None
            cap = circular_full_flow_lps(diam, slope, roughness)
            if cap:
                cap_total += cap

        rows.append({
            "node_id": nid,
            "dist_outfall_m": dist,
            "n_nodos_aguas_arriba": n_upstream,
            "q_pico_acum_base": q_acum,
            "upstream_capacity_lps": cap_total if cap_total > 0 else None,
        })

    topo_df = pd.DataFrame(rows)
    return static_df.merge(topo_df, on="node_id", how="left")
