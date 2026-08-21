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
    root = (
        Path(migration_dir)
        if migration_dir is not None
        else files("swmm_resilience.database").joinpath("sql")
    )
    entries = []
    invalid_sql_names = []
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        match = MIGRATION_NAME.fullmatch(entry.name)
        if match:
            entries.append((int(match.group(1)), match.group(2), entry))
        elif entry.name.lower().endswith(".sql"):
            invalid_sql_names.append(entry.name)
    if invalid_sql_names:
        names = ", ".join(sorted(invalid_sql_names))
        raise MigrationOrderError(f"Invalid migration filename(s): {names}")
    if not entries:
        raise MigrationOrderError(
            "Migration catalog is empty; expected a contiguous catalog from 001"
        )
    entries.sort(key=lambda item: (item[0], item[1]))
    versions = [item[0] for item in entries]
    if versions != list(range(1, len(entries) + 1)):
        raise MigrationOrderError(
            f"Migrations must be contiguous from 001: {versions}"
        )
    return entries


def _migration_catalog(migration_dir: Path | None):
    catalog = []
    for version, name, sql_path in _migration_entries(migration_dir):
        sql = sql_path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        catalog.append((version, name, sql, checksum))
    return catalog


def _applied_history(conn: sqlite3.Connection):
    exists = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name='schema_migrations'
        """
    ).fetchone()
    if not exists:
        return []
    return conn.execute(
        """
        SELECT version, name, checksum_sha256
        FROM schema_migrations
        ORDER BY version
        """
    ).fetchall()


def _validate_applied_history(applied, catalog) -> None:
    applied_versions = [row[0] for row in applied]
    expected_versions = list(range(1, len(applied) + 1))
    if applied_versions != expected_versions:
        raise MigrationOrderError(
            "Applied migrations must be a contiguous catalog prefix: "
            f"{applied_versions}"
        )
    if len(applied) > len(catalog):
        raise MigrationOrderError(
            "Applied migration history is ahead of the migration catalog; "
            "at least one applied migration file is missing"
        )
    for row, (version, name, _sql, checksum) in zip(applied, catalog):
        if row[0] != version:
            raise MigrationOrderError(
                "Applied migrations are not a prefix of the migration catalog"
            )
        if row[1] != name or row[2] != checksum:
            raise MigrationChecksumError(
                f"Applied migration {version:03d} identity/checksum differs"
            )


def _migration_statements(sql: str):
    buffer = []
    for character in sql:
        buffer.append(character)
        if character == ";" and sqlite3.complete_statement("".join(buffer)):
            statement = "".join(buffer).strip()
            if statement:
                yield statement
            buffer.clear()
    remainder = "".join(buffer).strip()
    if remainder:
        yield remainder


def _execute_migration_sql(conn: sqlite3.Connection, sql: str) -> None:
    denied_operation = []
    forbidden_actions = {
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_SAVEPOINT,
    }

    def deny_transaction_control(
        action_code,
        argument_one,
        argument_two,
        _database_name,
        _trigger_name,
    ):
        if action_code in forbidden_actions:
            denied_operation.append(argument_one or argument_two or "unknown")
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(deny_transaction_control)
    try:
        for statement in _migration_statements(sql):
            denied_operation.clear()
            try:
                conn.execute(statement)
            except sqlite3.DatabaseError as exc:
                if denied_operation:
                    operation = denied_operation[-1]
                    raise RuntimeError(
                        "Migration transaction control is forbidden: "
                        f"{operation}"
                    ) from exc
                raise
    finally:
        conn.set_authorizer(None)


def apply_migrations(
    conn: sqlite3.Connection,
    migration_dir: Path | None = None,
) -> None:
    if conn.in_transaction:
        raise RuntimeError(
            "Cannot apply migrations while the connection has an active transaction"
        )

    catalog = _migration_catalog(migration_dir)
    applied = _applied_history(conn)
    _validate_applied_history(applied, catalog)

    applied_any = False
    for version, name, sql, checksum in catalog[len(applied):]:
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute("BEGIN IMMEDIATE")
            _execute_migration_sql(conn, sql)
            conn.execute(
                """
                INSERT INTO schema_migrations (
                    version, name, checksum_sha256, applied_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (version, name, checksum, stamp),
            )
            conn.commit()
            applied_any = True
        except Exception:
            conn.rollback()
            raise

    if applied_any:
        conn.execute("PRAGMA optimize")
