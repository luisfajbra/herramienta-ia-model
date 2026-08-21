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
