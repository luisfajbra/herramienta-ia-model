import hashlib
from pathlib import Path
import sqlite3


def connect_database(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA recursive_triggers = ON")
    if conn.execute("PRAGMA recursive_triggers").fetchone()[0] != 1:
        conn.close()
        raise RuntimeError("SQLite recursive triggers could not be enabled")
    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if str(mode).lower() != "wal":
        conn.close()
        raise RuntimeError(f"SQLite WAL mode unavailable: {mode}")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _sha256_sql_function(value) -> str:
    if value is None:
        raise ValueError("sha256() requires a non-NULL argument")
    payload = value if isinstance(value, (bytes, bytearray)) else str(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def connect_managed_database(path: str | Path) -> sqlite3.Connection:
    conn = connect_database(path)
    conn.create_function("sha256", 1, _sha256_sql_function, deterministic=True)
    return conn
