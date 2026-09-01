from swmm_resilience.database.connection import connect_database


def test_connection_enables_safety_pragmas(tmp_path):
    conn = connect_database(tmp_path / "nested" / "test.sqlite3")
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA recursive_triggers").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("SELECT 7 AS value").fetchone()["value"] == 7
    finally:
        conn.close()


import hashlib

import pytest

from swmm_resilience.database.connection import connect_managed_database


def test_managed_connection_registers_sha256_function(tmp_path):
    conn = connect_managed_database(tmp_path / "db.sqlite3")
    try:
        digest = conn.execute("SELECT sha256(?)", (b"hello",)).fetchone()[0]
        assert digest == hashlib.sha256(b"hello").hexdigest()
        text_digest = conn.execute("SELECT sha256(?)", ("hello",)).fetchone()[0]
        assert text_digest == hashlib.sha256(b"hello").hexdigest()
    finally:
        conn.close()


def test_sha256_function_rejects_non_bytes_non_str_input(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database

    conn = connect_managed_database(tmp_path / "db.sqlite3")
    try:
        with pytest.raises(Exception):
            conn.execute("SELECT sha256(?)", (42,))
        with pytest.raises(Exception):
            conn.execute("SELECT sha256(?)", (3.14,))
    finally:
        conn.close()


def test_raw_connection_has_no_sha256_function(tmp_path):
    from swmm_resilience.database.connection import connect_database

    conn = connect_database(tmp_path / "db.sqlite3")
    try:
        with pytest.raises(Exception):
            conn.execute("SELECT sha256(?)", (b"hello",))
    finally:
        conn.close()
