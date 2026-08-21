from pathlib import Path
import sqlite3
from statistics import median
from time import perf_counter

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations


pytestmark = pytest.mark.scale

MILLION_ROWS = 1_000_000
BENCHMARK_ROWS = 200_000
BATCH_SIZE = 10_000
MAX_INSERT_SECONDS = 120
MAX_DATABASE_BYTES = 500 * 1024 * 1024

INSERT_TIMESERIES = """
    INSERT INTO node_timeseries (
        run_id, network_id, node_pk, step_index, time_sec,
        total_inflow_lps, lateral_inflow_lps, depth_m, depth_ratio,
        flooding_lps, total_outflow_lps, failed_now
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _seed_timeseries_parents(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO networks (
            network_id, network_sha256, name, source_filename, inp_bytes,
            flow_units, created_at_utc
        ) VALUES (1, ?, 'scale-network', 'scale.inp', ?, 'LPS', ?)
        """,
        (
            "1" * 64,
            b"scale fixture",
            "2026-08-21T00:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO nodes (
            node_pk, network_id, node_id, node_type,
            invert_elevation_m, max_depth_m, base_inflow_lps
        ) VALUES (1, 1, 'J1', 'junction', 100.0, 2.0, 0.5)
        """
    )
    conn.execute(
        """
        INSERT INTO scenarios (
            scenario_id, network_id, scenario_key, scenario_kind,
            factor_mult, shape_id, duracion_horas, tiempo_al_pico_h,
            config_json, config_sha256
        ) VALUES (1, 1, 'scale-scenario', 'factor', 1.0, 'uniform',
                  2.0, 1.0, '{}', ?)
        """,
        ("2" * 64,),
    )
    conn.execute(
        """
        INSERT INTO runs (
            run_id, scenario_id, network_id, status, config_sha256
        ) VALUES (1, 1, 1, 'COMPLETE', ?)
        """,
        ("3" * 64,),
    )
    conn.commit()


def _timeseries_rows(start: int, stop: int):
    for step_index in range(start, stop):
        total_inflow = float(step_index % 1_000) / 10.0
        depth = float(step_index % 200) / 100.0
        flooding = float(step_index % 50) / 20.0
        yield (
            1,
            1,
            1,
            step_index,
            float(step_index) * 60.0,
            total_inflow,
            total_inflow / 4.0,
            depth,
            depth / 2.0,
            flooding,
            total_inflow - flooding,
            0,
        )


def _insert_timeseries(conn: sqlite3.Connection, row_count: int) -> float:
    started = perf_counter()
    for start in range(0, row_count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, row_count)
        conn.executemany(
            INSERT_TIMESERIES,
            _timeseries_rows(start, stop),
        )
    conn.commit()
    return perf_counter() - started


def _checkpoint(conn: sqlite3.Connection) -> None:
    result = tuple(conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
    assert result[0] == 0, result


def _remove_sqlite_artifacts(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def test_million_temporal_rows_stay_within_foundation_guardrails(tmp_path):
    db_path = tmp_path / "million-temporal-rows.sqlite3"
    conn = None
    try:
        conn = connect_database(db_path)
        apply_migrations(conn)
        _seed_timeseries_parents(conn)

        elapsed_seconds = _insert_timeseries(conn, MILLION_ROWS)
        inserted_rows = conn.execute(
            "SELECT COUNT(*) FROM node_timeseries"
        ).fetchone()[0]
        _checkpoint(conn)
        database_bytes = db_path.stat().st_size

        print(
            "million_row_temporal_load "
            f"rows={inserted_rows} "
            f"seconds={elapsed_seconds:.3f} "
            f"database_bytes={database_bytes}"
        )

        assert inserted_rows == MILLION_ROWS
        assert elapsed_seconds < MAX_INSERT_SECONDS
        assert database_bytes < MAX_DATABASE_BYTES
    finally:
        if conn is not None:
            conn.close()
        _remove_sqlite_artifacts(db_path)


def _replace_timeseries_with_without_rowid(
    source_conn: sqlite3.Connection,
    target_conn: sqlite3.Connection,
) -> None:
    row = source_conn.execute(
        """
        SELECT sql
        FROM sqlite_schema
        WHERE type = 'table' AND name = 'node_timeseries'
        """
    ).fetchone()
    assert row is not None and row[0]
    rowid_table_sql = str(row[0]).rstrip().rstrip(";")
    assert "WITHOUT ROWID" not in rowid_table_sql.upper()

    target_conn.execute("DROP TABLE node_timeseries")
    target_conn.execute(f"{rowid_table_sql} WITHOUT ROWID")
    target_conn.execute(
        """
        CREATE INDEX idx_timeseries_run_time
        ON node_timeseries(run_id, time_sec)
        """
    )
    target_conn.commit()


def _secondary_index_definitions(
    conn: sqlite3.Connection,
) -> tuple[tuple[str, bool, bool, tuple[str, ...]], ...]:
    definitions = []
    for row in conn.execute("PRAGMA index_list('node_timeseries')"):
        name = str(row[1])
        if name.startswith("sqlite_autoindex_"):
            continue
        columns = tuple(
            str(column[2])
            for column in conn.execute(f"PRAGMA index_info('{name}')")
        )
        definitions.append((name, bool(row[2]), bool(row[4]), columns))
    return tuple(sorted(definitions))


def _column_definitions(
    conn: sqlite3.Connection,
) -> tuple[tuple[str, str, bool, object, int], ...]:
    rows = conn.execute("PRAGMA table_info('node_timeseries')").fetchall()
    return tuple(
        (
            str(row[1]),
            str(row[2]),
            bool(row[3]),
            row[4],
            int(row[5]),
        )
        for row in rows
    )


def _timeseries_identity(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
    row = conn.execute(
        """
        SELECT COUNT(*), MIN(step_index), MAX(step_index), SUM(step_index)
        FROM node_timeseries
        """
    ).fetchone()
    return tuple(int(value) for value in row)


def _timed_window_read(conn: sqlite3.Connection) -> tuple[float, int]:
    started = perf_counter()
    rows = conn.execute(
        """
        SELECT *
        FROM node_timeseries
        WHERE run_id = 1
          AND node_pk = 1
          AND step_index BETWEEN 90000 AND 109999
        ORDER BY step_index
        """
    ).fetchall()
    elapsed_ms = (perf_counter() - started) * 1_000.0
    return elapsed_ms, len(rows)


def test_rowid_and_without_rowid_temporal_layout_benchmark(tmp_path):
    rowid_path = tmp_path / "temporal-rowid.sqlite3"
    without_rowid_path = tmp_path / "temporal-without-rowid.sqlite3"
    rowid_conn = None
    without_rowid_conn = None
    try:
        rowid_conn = connect_database(rowid_path)
        without_rowid_conn = connect_database(without_rowid_path)
        apply_migrations(rowid_conn)
        apply_migrations(without_rowid_conn)
        _replace_timeseries_with_without_rowid(rowid_conn, without_rowid_conn)

        expected_indexes = {"idx_timeseries_run_time"}
        rowid_indexes = _secondary_index_definitions(rowid_conn)
        without_rowid_indexes = _secondary_index_definitions(
            without_rowid_conn
        )
        assert _column_definitions(rowid_conn) == _column_definitions(
            without_rowid_conn
        )
        primary_key = tuple(
            name
            for name, _type, _not_null, _default, pk_order in sorted(
                _column_definitions(rowid_conn),
                key=lambda definition: definition[4] or 99,
            )
            if pk_order
        )
        assert primary_key == ("run_id", "node_pk", "step_index")
        assert {
            definition[0] for definition in rowid_indexes
        } == expected_indexes
        assert rowid_indexes == without_rowid_indexes

        _seed_timeseries_parents(rowid_conn)
        _seed_timeseries_parents(without_rowid_conn)
        _insert_timeseries(rowid_conn, BENCHMARK_ROWS)
        _insert_timeseries(without_rowid_conn, BENCHMARK_ROWS)

        assert _timeseries_identity(rowid_conn) == (
            BENCHMARK_ROWS,
            0,
            BENCHMARK_ROWS - 1,
            BENCHMARK_ROWS * (BENCHMARK_ROWS - 1) // 2,
        )
        assert _timeseries_identity(rowid_conn) == _timeseries_identity(
            without_rowid_conn
        )

        _checkpoint(rowid_conn)
        _checkpoint(without_rowid_conn)
        rowid_bytes = rowid_path.stat().st_size
        without_rowid_bytes = without_rowid_path.stat().st_size

        assert _timed_window_read(rowid_conn)[1] == 20_000
        assert _timed_window_read(without_rowid_conn)[1] == 20_000

        timings = {"rowid": [], "without_rowid": []}
        connections = {
            "rowid": rowid_conn,
            "without_rowid": without_rowid_conn,
        }
        for iteration in range(5):
            order = (
                ("rowid", "without_rowid")
                if iteration % 2 == 0
                else ("without_rowid", "rowid")
            )
            for layout in order:
                elapsed_ms, returned_rows = _timed_window_read(
                    connections[layout]
                )
                assert returned_rows == 20_000
                timings[layout].append(elapsed_ms)

        rowid_median_ms = median(timings["rowid"])
        without_rowid_median_ms = median(timings["without_rowid"])
        print(
            "temporal_layout_benchmark "
            f"rows={BENCHMARK_ROWS} "
            f"rowid_median_ms={rowid_median_ms:.3f} "
            f"rowid_bytes={rowid_bytes} "
            f"without_rowid_median_ms={without_rowid_median_ms:.3f} "
            f"without_rowid_bytes={without_rowid_bytes}"
        )

        assert len(timings["rowid"]) == 5
        assert len(timings["without_rowid"]) == 5
    finally:
        if rowid_conn is not None:
            rowid_conn.close()
        if without_rowid_conn is not None:
            without_rowid_conn.close()
        _remove_sqlite_artifacts(rowid_path)
        _remove_sqlite_artifacts(without_rowid_path)
