"""
Dataset construction for machine learning.
"""

import sqlite3

import pandas as pd


def export_ml_dataset(db_file: str, output_csv: str, network_hash: str | None = None):
    """Export a flat dataset with static, dynamic and target variables."""
    conn = sqlite3.connect(db_file)
    query = """
        SELECT
            nr.run_id,
            nr.node_id,
            r.network_hash,
            r.network_file,
            r.inflow_multiplier,
            r.scenario_type,
            r.spatial_pattern,
            nn.invert_elev_m,
            nn.full_depth_m,
            nn.base_inflow_lps,
            nn.node_type,
            nn.in_degree,
            nn.out_degree,
            nn.upstream_pipes_count,
            nn.upstream_diam_max_m,
            nn.upstream_diam_min_m,
            nn.upstream_diam_avg_m,
            nn.upstream_slope_avg,
            nn.upstream_slope_max,
            nn.upstream_capacity_lps,
            nn.downstream_pipes_count,
            nn.downstream_diam_max_m,
            nn.downstream_diam_min_m,
            nn.downstream_diam_avg_m,
            nn.downstream_slope_avg,
            nn.downstream_slope_max,
            nn.downstream_capacity_lps,
            nr.max_depth_m,
            nr.max_depth_ratio,
            nr.time_to_peak_min,
            nr.depth_rate_m_per_min,
            nr.max_total_outflow_lps,
            nr.time_to_peak_outflow_min,
            nr.downstream_link_peak_flows_lps_json,
            nr.flooded,
            nr.flooding_volume_m3,
            nr.flooding_duration_min
        FROM node_results nr
        JOIN runs r ON nr.run_id = r.run_id
        LEFT JOIN network_nodes nn ON nn.node_uid = nr.node_id
                                  AND nn.network_hash = r.network_hash
        WHERE r.status = 'completed'
    """
    params: list[str] = []
    if network_hash is not None:
        query += " AND r.network_hash = ?"
        params.append(network_hash)
    query += " ORDER BY r.network_hash, r.scenario_type, r.inflow_multiplier, nr.node_id"
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    df.to_csv(output_csv, index=False)
    print(f"\nDataset ML exportado: {output_csv}  ({len(df)} filas, {len(df.columns)} columnas)")
    return df
