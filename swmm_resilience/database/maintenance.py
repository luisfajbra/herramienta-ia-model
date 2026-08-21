from pathlib import Path
import sqlite3


def optimize_database(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA optimize")


def checkpoint_and_backup(
    conn: sqlite3.Connection,
    destination: str | Path,
) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    busy, _log_frames, _checkpointed = conn.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    ).fetchone()
    if busy:
        raise RuntimeError(
            "Cannot back up SQLite database while WAL checkpoint is busy"
        )
    backup_conn = sqlite3.connect(target)
    try:
        backup_conn.execute("PRAGMA foreign_keys = ON")
        backup_conn.execute("PRAGMA recursive_triggers = ON")
        if backup_conn.execute("PRAGMA recursive_triggers").fetchone()[0] != 1:
            raise RuntimeError(
                "SQLite recursive triggers could not be enabled for backup"
            )
        conn.backup(backup_conn)
    finally:
        backup_conn.close()
    return target
