from datetime import datetime, timezone
import hashlib
from importlib.resources import files
from pathlib import Path
import re
import sqlite3


class MigrationChecksumError(RuntimeError):
    pass


class MigrationOrderError(RuntimeError):
    pass


MIGRATION_NAME = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")


def _migration_entries(migration_dir: Path | None):
    root = migration_dir or files("swmm_resilience.database").joinpath("sql")
    entries = []
    for entry in root.iterdir():
        match = MIGRATION_NAME.fullmatch(entry.name)
        if match:
            entries.append((int(match.group(1)), match.group(2), entry))
    entries.sort()
    versions = [item[0] for item in entries]
    if versions != list(range(1, len(entries) + 1)):
        raise MigrationOrderError(
            f"Migrations must be contiguous from 001: {versions}"
        )
    return entries


def apply_migrations(
    conn: sqlite3.Connection,
    migration_dir: Path | None = None,
) -> None:
    applied_any = False
    for version, name, sql_path in _migration_entries(migration_dir):
        sql = sql_path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        exists = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name='schema_migrations'
            """
        ).fetchone()
        row = None
        if exists:
            row = conn.execute(
                """
                SELECT name, checksum_sha256
                FROM schema_migrations
                WHERE version=?
                """,
                (version,),
            ).fetchone()
        if row:
            if row[0] != name or row[1] != checksum:
                raise MigrationChecksumError(
                    f"Applied migration {version:03d} identity/checksum differs"
                )
            continue

        stamp = datetime.now(timezone.utc).isoformat()
        record = (
            "INSERT INTO schema_migrations"
            "(version,name,checksum_sha256,applied_at_utc) VALUES"
            f"({version},'{name}','{checksum}','{stamp}');"
        )
        try:
            conn.executescript(f"BEGIN IMMEDIATE;\n{sql}\n{record}\nCOMMIT;")
            applied_any = True
        except Exception:
            conn.rollback()
            raise

    if applied_any:
        conn.execute("PRAGMA optimize")
