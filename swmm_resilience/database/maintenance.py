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
        conn.backup(backup_conn)
    finally:
        backup_conn.close()
    return target
