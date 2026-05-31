import sqlite3

import pandas as pd

from swmm_resilience.database.schema import create_schema


def test_run_summary_stores_total_flood_volume_m3():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    conn.execute(
        """
        INSERT INTO runs (
            run_id, network_file, network_hash, scenario_type, spatial_pattern,
            delta_inflow_lps, inflow_multiplier, status
        ) VALUES ('run-1', 'network.inp', 'hash-1', 'steady', 'uniform', 0.0, 2.0, 'completed')
        """
    )
    conn.execute(
        """
        INSERT INTO run_summary (
            summary_id, run_id, inflow_multiplier, total_nodes, failed_nodes_count,
            total_peak_flooding_lps, total_flood_volume_m3, pct_flooded_nodes,
            resilience_index
        ) VALUES ('sum-1', 'run-1', 2.0, 3, 1, 12.0, 3.5, 33.33, 0.6667)
        """
    )

    df = pd.read_sql(
        """
        SELECT s.total_flood_volume_m3
        FROM runs r
        LEFT JOIN run_summary s ON r.run_id = s.run_id
        WHERE r.status = 'completed'
        """,
        conn,
    )

    assert df.loc[0, "total_flood_volume_m3"] == 3.5
