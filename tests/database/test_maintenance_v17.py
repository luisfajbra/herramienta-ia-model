import sqlite3

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.maintenance import (
    checkpoint_and_backup,
    optimize_database,
)
from swmm_resilience.database.migrations import apply_migrations


def _insert_network(conn, network_id=1):
    conn.execute(
        """
        INSERT INTO networks (
            network_id, network_sha256, name, source_filename, inp_bytes,
            flow_units, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            network_id,
            f"{network_id:064x}",
            f"network-{network_id}",
            f"network-{network_id}.inp",
            b"inp",
            "LPS",
            "2026-08-21T00:00:00+00:00",
        ),
    )


def test_checkpoint_and_backup_creates_standalone_database(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "backups" / "snapshot.sqlite3"
    conn = connect_database(source)
    try:
        apply_migrations(conn)
        _insert_network(conn)
        conn.commit()

        assert checkpoint_and_backup(conn, destination) == destination

        backup_conn = sqlite3.connect(destination)
        try:
            assert (
                backup_conn.execute("PRAGMA integrity_check").fetchone()[0]
                == "ok"
            )
            assert backup_conn.execute(
                "SELECT COUNT(*) FROM schema_migrations"
            ).fetchone()[0] == 2
            assert backup_conn.execute(
                "SELECT COUNT(*) FROM networks"
            ).fetchone()[0] == 1
        finally:
            backup_conn.close()

        assert not destination.with_name(destination.name + "-wal").exists()
        assert not destination.with_name(destination.name + "-shm").exists()
    finally:
        conn.close()


def test_backup_enables_foreign_keys_on_destination_connection(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "snapshot.sqlite3"
    conn = connect_database(source)
    statements = []
    real_connect = sqlite3.connect

    def connect_with_trace(*args, **kwargs):
        traced_conn = real_connect(*args, **kwargs)
        traced_conn.set_trace_callback(statements.append)
        return traced_conn

    try:
        apply_migrations(conn)
        monkeypatch.setattr(
            "swmm_resilience.database.maintenance.sqlite3.connect",
            connect_with_trace,
        )

        checkpoint_and_backup(conn, destination)

        assert any(
            statement.strip().lower() == "pragma foreign_keys = on"
            for statement in statements
        )
    finally:
        conn.close()


def test_checkpoint_and_backup_rejects_busy_wal(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "snapshot.sqlite3"
    conn = connect_database(source)
    writer = None
    try:
        apply_migrations(conn)
        conn.execute("PRAGMA busy_timeout = 1")
        writer = sqlite3.connect(source, timeout=0.1)
        writer.execute("BEGIN IMMEDIATE")
        _insert_network(writer)

        with pytest.raises(RuntimeError, match="checkpoint is busy"):
            checkpoint_and_backup(conn, destination)
        assert not destination.exists()
    finally:
        if writer is not None:
            writer.rollback()
            writer.close()
        conn.close()


def test_optimize_database_executes_pragma_without_changing_rows(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    try:
        apply_migrations(conn)
        _insert_network(conn)
        conn.commit()
        statements = []
        conn.set_trace_callback(statements.append)

        optimize_database(conn)

        conn.set_trace_callback(None)
        assert "PRAGMA optimize" in statements
        assert conn.execute("SELECT name FROM networks").fetchone()[0] == "network-1"
    finally:
        conn.close()
