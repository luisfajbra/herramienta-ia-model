import pandas as pd
from pathlib import Path

_STATIC_COLS = [
    "node_id", "elev_fondo", "prof_max",
    "diam_max_in", "diam_max_out", "pendiente_max_in", "pendiente_out",
    "base_inflow_lps", "dist_outfall_m", "n_nodos_aguas_arriba",
    "q_pico_acum_base", "upstream_capacity_lps", "coord_x", "coord_y",
]


def assemble_dataset(
    static_topo_df: pd.DataFrame,
    simulation_results: list,
    output_path: Path,
) -> pd.DataFrame:
    """Join static+topo features with dynamic features and labels for every factor.

    simulation_results: list of (factor, dynamic_df, labels_df)
    Writes the dataset to output_path and returns it.
    """
    static_base = static_topo_df[[c for c in _STATIC_COLS if c in static_topo_df.columns]]
    all_rows = []
    for _, dynamic_df, labels_df in simulation_results:
        merged = static_base.merge(dynamic_df, on="node_id", how="left")
        required_label_cols = {"vol_inundacion_m3", "inunda"}
        if not required_label_cols.issubset(labels_df.columns):
            raise ValueError("labels_df debe incluir columnas vol_inundacion_m3 e inunda")
        merged = merged.merge(labels_df, on="node_id", how="left")
        merged["vol_inundacion_m3"] = merged["vol_inundacion_m3"].fillna(0.0)
        merged["inunda"] = merged["inunda"].fillna(0).astype(int)
        all_rows.append(merged)

    dataset = pd.concat(all_rows, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Dataset guardado: {output_path}  ({dataset.shape[0]} filas × {dataset.shape[1]} cols)")
    return dataset
