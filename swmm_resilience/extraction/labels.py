import pandas as pd
from pathlib import Path

from ..simulation.swmm_api_io import read_node_flooding_summary


def extract_labels(rpt_path: Path, all_node_ids: list, threshold_m3: float = 0.0) -> pd.DataFrame:
    """Parse Node Flooding Summary from a SWMM .rpt file.

    Nodes absent from the .rpt (no flooding reported) receive vol=0, inunda=0.
    read_node_flooding_summary already converts 10^6 L → m³ (×1000).

    Returns DataFrame: node_id, vol_inundacion_m3, inunda
    """
    result = pd.DataFrame({"node_id": [str(n) for n in all_node_ids]})
    result["vol_inundacion_m3"] = 0.0

    df_rpt = read_node_flooding_summary(rpt_path)
    if df_rpt is not None and not df_rpt.empty and "flooding_volume_m3" in df_rpt.columns:
        df_rpt["node_id"] = df_rpt["node_id"].astype(str)
        flood_map = dict(zip(df_rpt["node_id"], df_rpt["flooding_volume_m3"].fillna(0.0)))
        result["vol_inundacion_m3"] = result["node_id"].map(flood_map).fillna(0.0)

    result["inunda"] = (result["vol_inundacion_m3"] > threshold_m3).astype(int)
    return result
