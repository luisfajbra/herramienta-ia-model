# swmm_resilience/database/upgrade.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3

from .connection import connect_database
from .maintenance import checkpoint_and_backup
from .migrations import apply_migrations
from .workflow_lock import WorkflowLock


class UpgradeIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class UpgradeReceipt:
    source_path: Path
    backup_path: Path | None
    backup_sha256: str | None
    schema_version_before: int
    logical_fingerprint: str | None


def _current_schema_version(conn: sqlite3.Connection) -> int:
    exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name='schema_migrations'
        """
    ).fetchone()
    if not exists:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return row[0] or 0


def _latest_catalog_version() -> int:
    from .migrations import _migration_catalog

    return max(version for version, _name, _sql, _checksum in _migration_catalog(None))


def _logical_fingerprint(conn: sqlite3.Connection) -> str:
    schema_version = conn.execute("PRAGMA schema_version").fetchone()[0]
    checksums = conn.execute(
        "SELECT checksum_sha256 FROM schema_migrations ORDER BY version"
    ).fetchall()
    payload = f"{schema_version}:{[row[0] for row in checksums]}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upgrade_database_with_backup(
    database_path: str | Path,
    backup_dir: str | Path,
) -> UpgradeReceipt:
    source_path = Path(database_path)
    with WorkflowLock(source_path):
        conn = connect_database(source_path)
        try:
            schema_version_before = _current_schema_version(conn)
            if schema_version_before >= _latest_catalog_version():
                return UpgradeReceipt(
                    source_path=source_path,
                    backup_path=None,
                    backup_sha256=None,
                    schema_version_before=schema_version_before,
                    logical_fingerprint=None,
                )

            fingerprint_before = _logical_fingerprint(conn)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = (
                Path(backup_dir)
                / f"{source_path.stem}.v{schema_version_before}.{stamp}.sqlite3"
            )
            checkpoint_and_backup(conn, backup_path)

            backup_conn = sqlite3.connect(backup_path)
            try:
                integrity = backup_conn.execute("PRAGMA integrity_check").fetchall()
                if integrity != [("ok",)]:
                    raise UpgradeIntegrityError(
                        f"Backup integrity check failed: {integrity}"
                    )
            finally:
                backup_conn.close()
            backup_sha256 = _file_sha256(backup_path)

            fingerprint_after = _logical_fingerprint(conn)
            if fingerprint_after != fingerprint_before:
                raise UpgradeIntegrityError(
                    "Database changed between backup and migration; refusing to proceed"
                )

            apply_migrations(conn)

            return UpgradeReceipt(
                source_path=source_path,
                backup_path=backup_path,
                backup_sha256=backup_sha256,
                schema_version_before=schema_version_before,
                logical_fingerprint=fingerprint_after,
            )
        finally:
            conn.close()
