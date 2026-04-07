"""
Database schema only.
"""


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS network_nodes (
    node_uid            TEXT PRIMARY KEY,
    network_hash        TEXT NOT NULL,
    invert_elev_m       REAL,
    full_depth_m        REAL,
    node_type           TEXT,
    in_degree           INTEGER,
    out_degree          INTEGER,
    upstream_pipes_count        INTEGER,
    upstream_diam_max_m         REAL,
    upstream_diam_min_m         REAL,
    upstream_diam_avg_m         REAL,
    upstream_slope_avg          REAL,
    upstream_slope_max          REAL,
    upstream_capacity_m3ps      REAL,
    downstream_pipes_count      INTEGER,
    downstream_diam_max_m       REAL,
    downstream_diam_min_m       REAL,
    downstream_diam_avg_m       REAL,
    downstream_slope_avg        REAL,
    downstream_slope_max        REAL,
    downstream_capacity_m3ps    REAL
);

CREATE TABLE IF NOT EXISTS network_links (
    link_uid            TEXT PRIMARY KEY,
    network_hash        TEXT NOT NULL,
    inlet_node          TEXT,
    outlet_node         TEXT,
    link_type           TEXT,
    diameter_m          REAL,
    length_m            REAL,
    roughness           REAL,
    slope_m_per_m       REAL,
    full_flow_capacity_m3ps REAL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    network_file        TEXT NOT NULL,
    network_hash        TEXT NOT NULL,
    scenario_type       TEXT NOT NULL,
    spatial_pattern     TEXT NOT NULL,
    delta_inflow_m3ps   REAL NOT NULL,
    executed_at         TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS run_inputs (
    input_id            TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    node_uid            TEXT NOT NULL,
    applied_inflow_m3ps REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS node_results (
    result_id               TEXT PRIMARY KEY,
    run_id                  TEXT NOT NULL REFERENCES runs(run_id),
    node_id                 TEXT NOT NULL,
    flooded                 INTEGER NOT NULL DEFAULT 0,
    flooding_volume_m3      REAL,
    flooding_duration_min   REAL,
    max_depth_m             REAL,
    max_depth_ratio         REAL,
    time_to_peak_min        REAL,
    depth_rate_m_per_min    REAL
);

CREATE TABLE IF NOT EXISTS link_results (
    result_id           TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    link_id             TEXT NOT NULL,
    max_flow_m3ps       REAL,
    max_velocity_mps    REAL,
    max_depth_m         REAL,
    max_capacity_ratio  REAL,
    surcharged          INTEGER NOT NULL DEFAULT 0,
    time_full_flow_hrs  REAL
);

CREATE TABLE IF NOT EXISTS run_summary (
    summary_id                  TEXT PRIMARY KEY,
    run_id                      TEXT NOT NULL REFERENCES runs(run_id),
    total_nodes                 INTEGER,
    total_flooded_nodes         INTEGER,
    total_flooding_volume_m3    REAL,
    pct_flooded_nodes           REAL,
    time_to_first_flood_min     REAL,
    resilience_index            REAL
);
"""


def create_schema(conn):
    """Create all database tables."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
