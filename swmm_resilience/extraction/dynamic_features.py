import networkx as nx
import pandas as pd


def compute_dynamic_features(static_topo_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Compute per-simulation dynamic features for a given factor multiplier.

    Used by the training pipeline, where every scenario is a uniform scaling of
    the base hydrographs, so accumulated peaks scale linearly with the factor.

    Returns DataFrame with columns: node_id, factor_mult, q_pico_nodo,
    q_pico_acum_escalado. factor_mult is dataset metadata, not a model input
    (see trainer.FEATURE_COLS).
    """
    df = static_topo_df[["node_id", "base_inflow_lps", "q_pico_acum_base"]].copy()
    df["factor_mult"] = round(factor, 6)
    df["q_pico_nodo"] = df["base_inflow_lps"] * factor
    df["q_pico_acum_escalado"] = df["q_pico_acum_base"] * factor
    return df[["node_id", "factor_mult", "q_pico_nodo", "q_pico_acum_escalado"]]


def compute_scenario_dynamic_features(
    static_topo_df: pd.DataFrame,
    peak_map: dict[str, float],
    graph: "nx.DiGraph",
) -> pd.DataFrame:
    """Compute dynamic features from real per-node scenario peaks (lps).

    Used at inference time for arbitrary hydrographs, where no global factor
    exists:

    - q_pico_nodo: the node's own scenario peak (0.0 for junctions without a
      direct inflow, consistent with training where base_inflow=0 x factor=0).
    - q_pico_acum_escalado: sum of real scenario peaks over the node's graph
      ancestors plus itself (same aggregation rule as q_pico_acum_base).

    Returns DataFrame with columns: node_id, q_pico_nodo, q_pico_acum_escalado.
    """
    rows = []
    for nid in static_topo_df["node_id"].astype(str):
        own_peak = float(peak_map.get(nid, 0.0))
        ancestors = nx.ancestors(graph, nid) if graph.has_node(nid) else set()
        accumulated = sum(
            float(peak_map.get(str(n), 0.0)) for n in ancestors | {nid}
        )
        rows.append(
            {
                "node_id": nid,
                "q_pico_nodo": own_peak,
                "q_pico_acum_escalado": accumulated,
            }
        )
    return pd.DataFrame(rows)
