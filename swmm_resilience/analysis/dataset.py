"""
Dataset construction for machine learning.
"""

import sqlite3

import pandas as pd


def export_ml_dataset(db_file: str, output_csv: str):
    """Export a flat dataset with static, dynamic and target variables."""
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(
        """
        SELECT
            nr.run_id,
            nr.node_id,
            r.delta_inflow_m3ps,
            r.scenario_type,
            r.spatial_pattern,
            ri.applied_inflow_m3ps,
            nn.invert_elev_m,
            nn.full_depth_m,
            nn.node_type,
            nn.in_degree,
            nn.out_degree,
            nn.upstream_pipes_count,
            nn.upstream_diam_max_m,
            nn.upstream_diam_min_m,
            nn.upstream_diam_avg_m,
            nn.upstream_slope_avg,
            nn.upstream_slope_max,
            nn.upstream_capacity_m3ps,
            nn.downstream_pipes_count,
            nn.downstream_diam_max_m,
            nn.downstream_diam_min_m,
            nn.downstream_diam_avg_m,
            nn.downstream_slope_avg,
            nn.downstream_slope_max,
            nn.downstream_capacity_m3ps,
            nr.max_depth_m,
            nr.max_depth_ratio,
            nr.time_to_peak_min,
            nr.depth_rate_m_per_min,
            nr.flooded,
            nr.flooding_volume_m3,
            nr.flooding_duration_min
        FROM node_results nr
        JOIN runs r ON nr.run_id = r.run_id
        JOIN run_inputs ri ON ri.run_id = nr.run_id
                         AND ri.node_uid = nr.node_id
        LEFT JOIN network_nodes nn ON nn.node_uid = nr.node_id
        WHERE r.status = 'completed'
        ORDER BY r.delta_inflow_m3ps, nr.node_id
        """,
        conn,
    )
    conn.close()
    df.to_csv(output_csv, index=False)
    print(f"\nDataset ML exportado: {output_csv}  ({len(df)} filas, {len(df.columns)} columnas)")
    return df
