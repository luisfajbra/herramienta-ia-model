import networkx as nx
import pandas as pd


def compute_dynamic_features(
    static_topo_df: pd.DataFrame,
    factor: float,
    duracion_horas: float = 0.0,
    tiempo_al_pico_h: float = 0.0,
) -> pd.DataFrame:
    """Compute per-simulation dynamic features for a given factor multiplier.

    Used by the training pipeline. duracion_horas and tiempo_al_pico_h are
    scenario-level scalars (same value for every node in the same simulation).

    Returns DataFrame with columns: node_id, factor_mult, q_pico_nodo,
    q_pico_acum_escalado, duracion_horas, tiempo_al_pico_h. factor_mult is
    dataset metadata, not a model input (see trainer.FEATURE_COLS).
    """
    df = static_topo_df[["node_id", "base_inflow_lps", "q_pico_acum_base"]].copy()
    df["factor_mult"] = round(factor, 6)
    df["q_pico_nodo"] = df["base_inflow_lps"] * factor
    df["q_pico_acum_escalado"] = df["q_pico_acum_base"] * factor
    df["duracion_horas"] = duracion_horas
    df["tiempo_al_pico_h"] = tiempo_al_pico_h
    return df[
        ["node_id", "factor_mult", "q_pico_nodo", "q_pico_acum_escalado",
         "duracion_horas", "tiempo_al_pico_h"]
    ]


def compute_scenario_dynamic_features(
    static_topo_df: pd.DataFrame,
    peak_map: dict[str, float],
    graph: "nx.DiGraph",
    duracion_horas: float = 0.0,
    tiempo_al_pico_h: float = 0.0,
) -> pd.DataFrame:
    """Compute dynamic features from real per-node scenario peaks (lps).

    Used at inference time for arbitrary hydrographs, where no global factor
    exists. duracion_horas and tiempo_al_pico_h are scenario-level scalars
    computed by the caller from the hydrograph series.

    Returns DataFrame with columns: node_id, q_pico_nodo, q_pico_acum_escalado,
    duracion_horas, tiempo_al_pico_h.
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
                "duracion_horas": duracion_horas,
                "tiempo_al_pico_h": tiempo_al_pico_h,
            }
        )
    return pd.DataFrame(rows)
