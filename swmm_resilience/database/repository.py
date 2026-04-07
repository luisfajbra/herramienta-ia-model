"""
Database read/write functions, without simulation logic.
"""

import sqlite3

import pandas as pd

from .schema import create_schema


def connect_db(db_file: str) -> sqlite3.Connection:
    """Open a SQLite database and ensure schema exists."""
    conn = sqlite3.connect(db_file)
    create_schema(conn)
    return conn


def save_static_topology(conn: sqlite3.Connection, topo: dict):
    """Persist network_nodes and network_links records."""
    df_nodes = pd.DataFrame(topo["node_records"])
    for _, row in df_nodes.iterrows():
        conn.execute(
            """
            INSERT OR IGNORE INTO network_nodes VALUES (
                :node_uid, :network_hash,
                :invert_elev_m, :full_depth_m, :node_type,
                :in_degree, :out_degree,
                :upstream_pipes_count,
                :upstream_diam_max_m, :upstream_diam_min_m, :upstream_diam_avg_m,
                :upstream_slope_avg, :upstream_slope_max, :upstream_capacity_m3ps,
                :downstream_pipes_count,
                :downstream_diam_max_m, :downstream_diam_min_m, :downstream_diam_avg_m,
                :downstream_slope_avg, :downstream_slope_max, :downstream_capacity_m3ps
            )
            """,
            dict(row),
        )

    df_links = pd.DataFrame(topo["link_records"])
    for _, row in df_links.iterrows():
        conn.execute(
            """
            INSERT OR IGNORE INTO network_links VALUES (
                :link_uid, :network_hash,
                :inlet_node, :outlet_node, :link_type,
                :diameter_m, :length_m, :roughness,
                :slope_m_per_m, :full_flow_capacity_m3ps
            )
            """,
            dict(row),
        )

    conn.commit()
    print(f"  [topo] {len(df_nodes)} nodos / {len(df_links)} links estaticos guardados.")


def save_results(conn: sqlite3.Connection, run_id: str, results: dict):
    """Save inputs, node results, link results and summary for one run."""
    df_inputs = pd.DataFrame(results["run_inputs"])
    df_inputs["run_id"] = run_id
    df_inputs.to_sql("run_inputs", conn, if_exists="append", index=False)

    df_nodes = pd.DataFrame(results["node_records"])
    df_nodes["run_id"] = run_id
    df_nodes.to_sql("node_results", conn, if_exists="append", index=False)

    df_links = pd.DataFrame(results["link_records"])
    df_links["run_id"] = run_id
    df_links.to_sql("link_results", conn, if_exists="append", index=False)

    df_summary = pd.DataFrame([results["summary"]])
    df_summary["run_id"] = run_id
    df_summary.to_sql("run_summary", conn, if_exists="append", index=False)

    conn.commit()


def update_run_status(conn: sqlite3.Connection, run_id: str, status: str):
    """Update run status."""
    conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))
    conn.commit()


def verify_run_saved(conn: sqlite3.Connection, run_id: str) -> dict:
    """Return counts per table for one run."""
    cursor = conn.cursor()

    def count_rows(table_name):
        return cursor.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]

    return {
        table_name: count_rows(table_name)
        for table_name in ["runs", "run_inputs", "node_results", "link_results", "run_summary"]
    }


def export_run_summary(db_file: str):
    """Print a compact summary of executed runs."""
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(
        """
        SELECT
            r.delta_inflow_m3ps,
            r.status,
            s.total_nodes,
            s.total_flooded_nodes,
            s.pct_flooded_nodes,
            s.total_flooding_volume_m3,
            s.resilience_index,
            s.time_to_first_flood_min
        FROM runs r
        LEFT JOIN run_summary s ON r.run_id = s.run_id
        ORDER BY r.delta_inflow_m3ps
        """,
        conn,
    )
    conn.close()
    print("\nResumen de corridas:")
    print(df.to_string(index=False))
