"""
Database schema only.
"""


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS network_nodes (
    network_hash        TEXT NOT NULL,
    node_uid            TEXT NOT NULL,
    invert_elev_m       REAL,
    full_depth_m        REAL,
    base_inflow_lps     REAL,
    node_type           TEXT,
    in_degree           INTEGER,
    out_degree          INTEGER,
    upstream_pipes_count        INTEGER,
    upstream_diam_max_m         REAL,
    upstream_diam_min_m         REAL,
    upstream_diam_avg_m         REAL,
    upstream_slope_avg          REAL,
    upstream_slope_max          REAL,
    upstream_capacity_lps       REAL,
    downstream_pipes_count      INTEGER,
    downstream_diam_max_m       REAL,
    downstream_diam_min_m       REAL,
    downstream_diam_avg_m       REAL,
    downstream_slope_avg        REAL,
    downstream_slope_max        REAL,
    downstream_capacity_lps     REAL,
    PRIMARY KEY (network_hash, node_uid)
);

CREATE TABLE IF NOT EXISTS network_links (
    network_hash        TEXT NOT NULL,
    link_uid            TEXT NOT NULL,
    inlet_node          TEXT,
    outlet_node         TEXT,
    link_type           TEXT,
    diameter_m          REAL,
    length_m            REAL,
    roughness           REAL,
    slope_m_per_m       REAL,
    full_flow_capacity_lps  REAL,
    PRIMARY KEY (network_hash, link_uid)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    network_file        TEXT NOT NULL,
    network_hash        TEXT NOT NULL,
    scenario_type       TEXT NOT NULL,
    spatial_pattern     TEXT NOT NULL,
    delta_inflow_lps    REAL NOT NULL,
    inflow_multiplier   REAL NOT NULL DEFAULT 1,
    executed_at         TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS run_inputs (
    input_id            TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    delta_inflow_lps    REAL NOT NULL,
    inflow_multiplier   REAL NOT NULL DEFAULT 1,
    node_uid            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS node_results (
    result_id               TEXT PRIMARY KEY,
    run_id                  TEXT NOT NULL REFERENCES runs(run_id),
    delta_inflow_lps        REAL NOT NULL,
    inflow_multiplier       REAL NOT NULL DEFAULT 1,
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
    delta_inflow_lps    REAL NOT NULL,
    inflow_multiplier   REAL NOT NULL DEFAULT 1,
    link_id             TEXT NOT NULL,
    max_flow_lps        REAL,
    max_velocity_mps    REAL,
    max_depth_m         REAL,
    max_capacity_ratio  REAL,
    surcharged          INTEGER NOT NULL DEFAULT 0,
    time_full_flow_hrs  REAL
);

CREATE TABLE IF NOT EXISTS run_summary (
    summary_id                  TEXT PRIMARY KEY,
    run_id                      TEXT NOT NULL REFERENCES runs(run_id),
    delta_inflow_lps            REAL NOT NULL,
    inflow_multiplier           REAL NOT NULL DEFAULT 1,
    total_nodes                 INTEGER,
    total_flooded_nodes         INTEGER,
    total_flooding_volume_m3    REAL,
    pct_flooded_nodes           REAL,
    time_to_first_flood_min     REAL,
    resilience_index            REAL
);
"""


REQUIRED_COLUMNS = {
    "network_nodes": {
        "base_inflow_lps": "REAL"
    },
    "runs": {
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1"
    },
    "run_inputs": {
        "delta_inflow_lps": "REAL NOT NULL DEFAULT 0",
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1"
    },
    "node_results": {
        "delta_inflow_lps": "REAL NOT NULL DEFAULT 0",
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1"
    },
    "link_results": {
        "delta_inflow_lps": "REAL NOT NULL DEFAULT 0",
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1"
    },
    "run_summary": {
        "delta_inflow_lps": "REAL NOT NULL DEFAULT 0",
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1"
    },
}


def _migrate_legacy_run_inputs(conn):
    """Drop the legacy applied_inflow_lps column while preserving existing rows."""
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(run_inputs)").fetchall()
    }
    if "applied_inflow_lps" not in columns:
        return

    conn.executescript(
        """
        ALTER TABLE run_inputs RENAME TO run_inputs_legacy;

        CREATE TABLE run_inputs (
            input_id            TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL REFERENCES runs(run_id),
            delta_inflow_lps    REAL NOT NULL,
            inflow_multiplier   REAL NOT NULL DEFAULT 1,
            node_uid            TEXT NOT NULL
        );

        INSERT INTO run_inputs (input_id, run_id, delta_inflow_lps, inflow_multiplier, node_uid)
        SELECT input_id, run_id, delta_inflow_lps, 1.0, node_uid
        FROM run_inputs_legacy;

        DROP TABLE run_inputs_legacy;
        """
    )


def create_schema(conn):
    """Create all database tables."""
    conn.executescript(SCHEMA_SQL)
    run_scoped_tables = {"run_inputs", "node_results", "link_results", "run_summary"}
    for table_name, columns in REQUIRED_COLUMNS.items():
        existing = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, column_sql in columns.items():
            if column_name not in existing:
                conn.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
                )
        if table_name in run_scoped_tables:
            conn.execute(
                f"""
                UPDATE {table_name}
                SET delta_inflow_lps = (
                    SELECT runs.delta_inflow_lps
                    FROM runs
                    WHERE runs.run_id = {table_name}.run_id
                )
                WHERE run_id IN (SELECT run_id FROM runs)
                  AND (delta_inflow_lps IS NULL OR delta_inflow_lps = 0)
                """
            )
            conn.execute(
                f"""
                UPDATE {table_name}
                SET inflow_multiplier = (
                    SELECT runs.inflow_multiplier
                    FROM runs
                    WHERE runs.run_id = {table_name}.run_id
                )
                WHERE run_id IN (SELECT run_id FROM runs)
                  AND (inflow_multiplier IS NULL OR inflow_multiplier = 0)
                """
            )
    _migrate_legacy_run_inputs(conn)
    conn.commit()
