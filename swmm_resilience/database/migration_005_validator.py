from __future__ import annotations

import sqlite3

from .migrations import MigrationPreflightError


def _foreign_key_violations(conn: sqlite3.Connection):
    return conn.execute("PRAGMA foreign_key_check").fetchall()


def _integrity_violations(conn: sqlite3.Connection):
    rows = [tuple(row) for row in conn.execute("PRAGMA integrity_check").fetchall()]
    return [] if rows == [("ok",)] else rows


def validate_before_005(conn: sqlite3.Connection) -> None:
    fk_violations = _foreign_key_violations(conn)
    if fk_violations:
        raise MigrationPreflightError(
            f"Migration 005 preflight: {len(fk_violations)} foreign-key "
            "violation(s) exist before upgrade; refusing to proceed"
        )

    integrity_violations = _integrity_violations(conn)
    if integrity_violations:
        raise MigrationPreflightError(
            f"Migration 005 preflight: {len(integrity_violations)} "
            "integrity_check violation(s) exist before upgrade; refusing to proceed"
        )

    try:
        conn.execute("SELECT training_run_id FROM training_runs").fetchall()
        conn.execute("SELECT promotion_id FROM model_promotions").fetchall()
    except sqlite3.DatabaseError as exc:
        raise MigrationPreflightError(
            "Migration 005 preflight: cannot enumerate pre-005 training runs "
            "and promotions for deterministic invalidation"
        ) from exc
