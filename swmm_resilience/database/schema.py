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
    executed_at         TEXT NOT NULL DEFAULT (datetime('now')),
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
    peak_flooding_lps       REAL,
    total_flood_volume_m3   REAL,
    flooding_duration_min   REAL,
    max_depth_m             REAL,
    max_depth_ratio         REAL,
    time_to_peak_min        REAL,
    depth_rate_m_per_min    REAL,
    max_total_outflow_lps   REAL,
    time_to_peak_outflow_min REAL,
    downstream_link_peak_flows_lps_json TEXT
);

CREATE TABLE IF NOT EXISTS link_results (
    result_id           TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    delta_inflow_lps    REAL,
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
    inflow_multiplier           REAL NOT NULL DEFAULT 1,
    total_nodes                 INTEGER,
    failed_nodes_count          INTEGER,
    total_peak_flooding_lps     REAL,
    total_flood_volume_m3       REAL,
    pct_flooded_nodes           REAL,
    time_to_first_flood_min     REAL,
    resilience_index            REAL
);

CREATE TABLE IF NOT EXISTS temporal_artifacts (
    artifact_id     TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    network_hash    TEXT NOT NULL,
    parquet_path    TEXT NOT NULL,
    node_count      INTEGER NOT NULL DEFAULT 0,
    step_count      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


REQUIRED_COLUMNS = {
    "network_nodes": {
        "base_inflow_lps": "REAL"
    },
    "runs": {
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1",
        "input_source": "TEXT NOT NULL DEFAULT 'steady'",
    },
    "run_inputs": {
        "delta_inflow_lps": "REAL NOT NULL DEFAULT 0",
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1"
    },
    "node_results": {
        "delta_inflow_lps": "REAL NOT NULL DEFAULT 0",
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1",
        "peak_flooding_lps": "REAL",
        "max_total_outflow_lps": "REAL",
        "time_to_peak_outflow_min": "REAL",
        "downstream_link_peak_flows_lps_json": "TEXT",
        "total_flood_volume_m3": "REAL",
    },
    "link_results": {
        "delta_inflow_lps": "REAL",
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1"
    },
    "run_summary": {
        "inflow_multiplier": "REAL NOT NULL DEFAULT 1",
        "failed_nodes_count": "INTEGER",
        "total_flood_volume_m3": "REAL",
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


def _migrate_run_summary(conn):
    """Simplify run_summary to one scenario column and one failed-nodes column."""
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(run_summary)").fetchall()
    }
    target_columns = {
        "summary_id",
        "run_id",
        "inflow_multiplier",
        "total_nodes",
        "failed_nodes_count",
        "total_peak_flooding_lps",
        "total_flood_volume_m3",
        "pct_flooded_nodes",
        "time_to_first_flood_min",
        "resilience_index",
    }
    if columns == target_columns:
        return

    inflow_expr = "COALESCE(inflow_multiplier, 1.0)" if "inflow_multiplier" in columns else "1.0"
    if "failed_nodes_count" in columns:
        failed_expr = "COALESCE(failed_nodes_count, 0)"
    elif "total_flooded_nodes" in columns:
        failed_expr = "COALESCE(total_flooded_nodes, 0)"
    else:
        failed_expr = "0"

    if "total_peak_flooding_lps" in columns:
        flood_expr = "total_peak_flooding_lps"
    else:
        flood_expr = "NULL"

    if "total_flood_volume_m3" in columns:
        volume_expr = "total_flood_volume_m3"
    elif "total_flooding_volume_m3" in columns:
        volume_expr = "total_flooding_volume_m3"
    else:
        volume_expr = "NULL"

    conn.executescript(
        f"""
        ALTER TABLE run_summary RENAME TO run_summary_legacy;

        CREATE TABLE run_summary (
            summary_id                  TEXT PRIMARY KEY,
            run_id                      TEXT NOT NULL REFERENCES runs(run_id),
            inflow_multiplier           REAL NOT NULL DEFAULT 1,
            total_nodes                 INTEGER,
            failed_nodes_count          INTEGER,
            total_peak_flooding_lps     REAL,
            total_flood_volume_m3       REAL,
            pct_flooded_nodes           REAL,
            time_to_first_flood_min     REAL,
            resilience_index            REAL
        );

        INSERT INTO run_summary (
            summary_id,
            run_id,
            inflow_multiplier,
            total_nodes,
            failed_nodes_count,
            total_peak_flooding_lps,
            total_flood_volume_m3,
            pct_flooded_nodes,
            time_to_first_flood_min,
            resilience_index
        )
        SELECT
            summary_id,
            run_id,
            {inflow_expr},
            total_nodes,
            {failed_expr},
            {flood_expr},
            {volume_expr},
            pct_flooded_nodes,
            time_to_first_flood_min,
            resilience_index
        FROM run_summary_legacy;

        DROP TABLE run_summary_legacy;
        """
    )


def _migrate_link_results_nullable_delta(conn):
    """Remove NOT NULL from link_results.delta_inflow_lps.

    Links (conduits) don't have direct inflows so the column is meaningless
    for them; NULL is the correct representation.  SQLite doesn't support
    ALTER COLUMN, so we recreate the table when the constraint is still present.
    """
    info = conn.execute("PRAGMA table_info(link_results)").fetchall()
    # row layout: (cid, name, type, notnull, dflt_value, pk)
    needs_migration = any(row[1] == "delta_inflow_lps" and row[3] == 1 for row in info)
    if not needs_migration:
        return

    # Build the INSERT using only columns that exist in the legacy table so
    # the migration works regardless of how many columns the old table has.
    legacy_cols = [row[1] for row in info]
    new_cols = [
        "result_id", "run_id", "delta_inflow_lps", "inflow_multiplier", "link_id",
        "max_flow_lps", "max_velocity_mps", "max_depth_m", "max_capacity_ratio",
        "surcharged", "time_full_flow_hrs",
    ]
    shared_cols = [c for c in new_cols if c in legacy_cols]
    col_list = ", ".join(shared_cols)

    conn.executescript(
        f"""
        ALTER TABLE link_results RENAME TO link_results_legacy;

        CREATE TABLE link_results (
            result_id           TEXT PRIMARY KEY,
            run_id              TEXT NOT NULL REFERENCES runs(run_id),
            delta_inflow_lps    REAL,
            inflow_multiplier   REAL NOT NULL DEFAULT 1,
            link_id             TEXT NOT NULL,
            max_flow_lps        REAL,
            max_velocity_mps    REAL,
            max_depth_m         REAL,
            max_capacity_ratio  REAL,
            surcharged          INTEGER NOT NULL DEFAULT 0,
            time_full_flow_hrs  REAL
        );

        INSERT INTO link_results ({col_list})
        SELECT {col_list} FROM link_results_legacy;

        DROP TABLE link_results_legacy;
        """
    )


def _migrate_legacy_node_flooding_volume(conn):
    """Preserve legacy volume columns without confusing them with peak flow."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(node_results)").fetchall()}
    if "flooding_volume_m3" in cols and "total_flood_volume_m3" not in cols:
        conn.execute("ALTER TABLE node_results RENAME COLUMN flooding_volume_m3 TO total_flood_volume_m3")
        conn.commit()


def create_schema(conn):
    """Create all database tables."""
    conn.executescript(SCHEMA_SQL)
    _migrate_legacy_run_inputs(conn)
    _migrate_run_summary(conn)
    _migrate_link_results_nullable_delta(conn)
    _migrate_legacy_node_flooding_volume(conn)
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
        existing = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if table_name in run_scoped_tables:
            if "delta_inflow_lps" in existing:
                # Only backfill rows that are truly NULL (column was added to a legacy DB
                # that already had rows). Never treat 0 as missing: for the
                # embedded_inflow_multiplier_sweep scenario, 0 is the correct delta for
                # nodes with no base inflow. Copying runs.delta_inflow_lps here would
                # replace a valid physical quantity with the run-level multiplier.
                conn.execute(
                    f"""
                    UPDATE {table_name}
                    SET delta_inflow_lps = (
                        SELECT runs.delta_inflow_lps
                        FROM runs
                        WHERE runs.run_id = {table_name}.run_id
                    )
                    WHERE run_id IN (SELECT run_id FROM runs)
                      AND delta_inflow_lps IS NULL
                    """
                )
            if "inflow_multiplier" in existing:
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
    conn.commit()
