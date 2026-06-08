import pandas as pd


def compute_dynamic_features(static_topo_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Compute per-simulation dynamic features for a given factor multiplier.

    Returns DataFrame with columns: node_id, factor_mult, q_pico_nodo, q_pico_acum_escalado
    """
    df = static_topo_df[["node_id", "base_inflow_lps", "q_pico_acum_base"]].copy()
    df["factor_mult"] = round(factor, 6)
    df["q_pico_nodo"] = df["base_inflow_lps"] * factor
    df["q_pico_acum_escalado"] = df["q_pico_acum_base"] * factor
    return df[["node_id", "factor_mult", "q_pico_nodo", "q_pico_acum_escalado"]]
