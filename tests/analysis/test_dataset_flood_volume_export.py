import sqlite3

import pandas as pd

from swmm_resilience.analysis.dataset import export_ml_dataset
from swmm_resilience.database.schema import create_schema


def test_export_ml_dataset_includes_volume_target_and_peak_feature(tmp_path):
    db_path = tmp_path / "runs.db"
    csv_path = tmp_path / "dataset.csv"
    conn = sqlite3.connect(db_path)
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
        INSERT INTO network_nodes (
            network_hash, node_uid, invert_elev_m, full_depth_m, base_inflow_lps,
            node_type, in_degree, out_degree
        ) VALUES ('hash-1', 'J1', 100.0, 2.0, 1.5, 'junction', 1, 1)
        """
    )
    conn.execute(
        """
        INSERT INTO node_results (
            result_id, run_id, delta_inflow_lps, inflow_multiplier, node_id,
            flooded, peak_flooding_lps, total_flood_volume_m3, flooding_duration_min
        ) VALUES ('res-1', 'run-1', 0.0, 2.0, 'J1', 1, 12.0, 3.5, 4.0)
        """
    )
    conn.commit()
    conn.close()

    df = export_ml_dataset(str(db_path), str(csv_path))

    assert "total_flood_volume_m3" in df.columns
    assert "peak_flooding_lps" in df.columns
    assert df.loc[0, "total_flood_volume_m3"] == 3.5
    assert df.loc[0, "peak_flooding_lps"] == 12.0
    assert pd.read_csv(csv_path).loc[0, "total_flood_volume_m3"] == 3.5
