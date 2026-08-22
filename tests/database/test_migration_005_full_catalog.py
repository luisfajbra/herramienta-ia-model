from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations


def test_005_applies_cleanly_from_a_fresh_database(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)  # packaged catalog: 001..005
    applied = [row[0] for row in conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()]
    assert applied == [1, 2, 3, 4, 5]


def test_005_is_idempotent(tmp_path):
    conn = connect_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    apply_migrations(conn)  # second call must be a no-op, not an error
    applied = [row[0] for row in conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()]
    assert applied == [1, 2, 3, 4, 5]
