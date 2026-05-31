import sqlite3

from swmm_resilience.database.schema import create_schema


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_schema_has_total_flood_volume_columns():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)

    assert "total_flood_volume_m3" in _columns(conn, "node_results")
    assert "total_flood_volume_m3" in _columns(conn, "run_summary")


def test_legacy_peak_column_is_not_renamed_to_volume_target():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE node_results (
            result_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            delta_inflow_lps REAL NOT NULL,
            inflow_multiplier REAL NOT NULL DEFAULT 1,
            node_id TEXT NOT NULL,
            flooded INTEGER NOT NULL DEFAULT 0,
            peak_flooding_lps REAL
        );
        """
    )

    create_schema(conn)
    columns = _columns(conn, "node_results")

    assert "peak_flooding_lps" in columns
    assert "total_flood_volume_m3" in columns


def test_legacy_node_flooding_volume_moves_to_total_volume_and_adds_peak_column():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE node_results (
            result_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            delta_inflow_lps REAL NOT NULL,
            inflow_multiplier REAL NOT NULL DEFAULT 1,
            node_id TEXT NOT NULL,
            flooded INTEGER NOT NULL DEFAULT 0,
            flooding_volume_m3 REAL
        );

        INSERT INTO node_results (
            result_id,
            run_id,
            delta_inflow_lps,
            inflow_multiplier,
            node_id,
            flooded,
            flooding_volume_m3
        )
        VALUES ('result-1', 'run-1', 0.0, 1.0, 'J1', 1, 12.5);
        """
    )

    create_schema(conn)
    columns = _columns(conn, "node_results")
    row = conn.execute(
        """
        SELECT total_flood_volume_m3, peak_flooding_lps
        FROM node_results
        WHERE result_id = 'result-1'
        """
    ).fetchone()

    assert "total_flood_volume_m3" in columns
    assert "peak_flooding_lps" in columns
    assert row == (12.5, None)


def test_legacy_run_summary_preserves_peak_and_volume_separately():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE run_summary (
            summary_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            inflow_multiplier REAL NOT NULL DEFAULT 1,
            total_nodes INTEGER,
            failed_nodes_count INTEGER,
            total_peak_flooding_lps REAL,
            total_flooding_volume_m3 REAL,
            pct_flooded_nodes REAL,
            time_to_first_flood_min REAL,
            resilience_index REAL
        );

        INSERT INTO run_summary (
            summary_id,
            run_id,
            inflow_multiplier,
            total_nodes,
            failed_nodes_count,
            total_peak_flooding_lps,
            total_flooding_volume_m3,
            pct_flooded_nodes,
            time_to_first_flood_min,
            resilience_index
        )
        VALUES ('summary-1', 'run-1', 1.0, 2, 1, 7.5, 42.25, 50.0, 3.0, 0.8);
        """
    )

    create_schema(conn)
    row = conn.execute(
        """
        SELECT total_peak_flooding_lps, total_flood_volume_m3
        FROM run_summary
        WHERE summary_id = 'summary-1'
        """
    ).fetchone()

    assert row == (7.5, 42.25)
