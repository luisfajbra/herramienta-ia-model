"""Register and verify immutable links to the files behind a completed run."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


REQUIRED_ARTIFACT_ROLES = frozenset(
    {"effective_inp", "event_source", "swmm_rpt", "swmm_out"}
)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _artifact_items(artifacts) -> list[tuple[object, object]]:
    if isinstance(artifacts, Mapping):
        return list(artifacts.items())
    if isinstance(artifacts, Iterable) and not isinstance(artifacts, (str, bytes)):
        try:
            return [tuple(item) for item in artifacts]
        except (TypeError, ValueError) as exc:
            raise ValueError("artifacts must contain role/path pairs") from exc
    raise ValueError("artifacts must be a role-to-path mapping or role/path pairs")


def _prepare_artifacts(artifacts) -> dict[str, tuple[str, str, int]]:
    prepared = {}
    for item in _artifact_items(artifacts):
        if len(item) != 2:
            raise ValueError("artifacts must contain role/path pairs")
        role, source_path = item
        if not isinstance(role, str) or role not in REQUIRED_ARTIFACT_ROLES:
            raise ValueError(f"unknown artifact role: {role!r}")
        if role in prepared:
            raise ValueError(f"duplicate artifact role: {role}")
        try:
            path = Path(source_path).resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"artifact path for {role} does not resolve: {source_path!r}") from exc
        if not path.is_file():
            raise ValueError(f"artifact for {role} is not a file: {path}")
        try:
            payload = path.read_bytes()
            byte_size = len(payload)
        except OSError as exc:
            raise ValueError(f"artifact for {role} cannot be read: {path}") from exc
        if byte_size <= 0:
            raise ValueError(f"artifact for {role} is empty: {path}")
        prepared[role] = (str(path), hashlib.sha256(payload).hexdigest(), byte_size)

    missing = REQUIRED_ARTIFACT_ROLES - prepared.keys()
    if missing:
        raise ValueError(f"missing artifact role(s): {', '.join(sorted(missing))}")
    return prepared


def _run_report_sha256(conn: sqlite3.Connection, run_id: int) -> str:
    run = conn.execute(
        "SELECT status FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run is None:
        raise ValueError(f"run_id does not exist: {run_id}")
    if run[0] != "COMPLETE":
        raise ValueError("source artifacts require a COMPLETE run")
    quality = conn.execute(
        "SELECT manifest_json FROM run_hydraulic_quality WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if quality is None:
        raise ValueError("source artifacts require immutable hydraulic quality")
    try:
        manifest = json.loads(quality[0])
        report_sha256 = manifest["flooding_report"]["report_sha256"]
    except (TypeError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError("hydraulic quality manifest has no flooding report hash") from exc
    if not _valid_sha256(report_sha256):
        raise ValueError("hydraulic quality manifest has an invalid flooding report hash")
    return report_sha256


def _load_records(conn: sqlite3.Connection, run_id: int) -> dict[str, dict]:
    rows = conn.execute(
        """SELECT artifact_role, resolved_path, sha256, byte_size,
                  source_manifest_sha256, registered_at_utc
           FROM run_source_artifacts WHERE run_id = ? ORDER BY artifact_role""",
        (run_id,),
    ).fetchall()
    return {
        row[0]: {
            "artifact_role": row[0],
            "resolved_path": row[1],
            "sha256": row[2],
            "byte_size": row[3],
            "source_manifest_sha256": row[4],
            "registered_at_utc": row[5],
        }
        for row in rows
    }


def register_run_source_artifacts(
    conn: sqlite3.Connection,
    run_id: int,
    artifacts,
    source_manifest_sha256: str,
) -> dict[str, dict]:
    """Atomically register one complete role set for a COMPLETE, QA-assessed run."""
    if not _valid_sha256(source_manifest_sha256):
        raise ValueError("source_manifest_sha256 must be lowercase 64-hex SHA-256")
    try:
        run_id = int(run_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("run_id must be an integer") from exc
    prepared = _prepare_artifacts(artifacts)
    report_sha256 = _run_report_sha256(conn, run_id)
    if prepared["swmm_rpt"][1] != report_sha256:
        raise ValueError("SWMM RPT hash does not match hydraulic quality manifest")
    if conn.execute(
        "SELECT 1 FROM run_source_artifacts WHERE run_id = ? LIMIT 1", (run_id,)
    ).fetchone():
        raise ValueError(f"source artifacts already registered for run_id {run_id}")

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    conn.execute("SAVEPOINT register_run_source_artifacts")
    try:
        for role in sorted(REQUIRED_ARTIFACT_ROLES):
            path, sha256, byte_size = prepared[role]
            conn.execute(
                """INSERT INTO run_source_artifacts (
                       run_id, artifact_role, resolved_path, sha256, byte_size,
                       source_manifest_sha256, registered_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, role, path, sha256, byte_size, source_manifest_sha256, timestamp),
            )
        conn.execute("RELEASE SAVEPOINT register_run_source_artifacts")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT register_run_source_artifacts")
        conn.execute("RELEASE SAVEPOINT register_run_source_artifacts")
        raise
    return _load_records(conn, run_id)


def verify_run_source_artifacts(conn: sqlite3.Connection, run_id: int) -> dict[str, dict]:
    """Verify the complete registered role set against the current source files."""
    try:
        run_id = int(run_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("run_id must be an integer") from exc
    records = _load_records(conn, run_id)
    if records.keys() != REQUIRED_ARTIFACT_ROLES:
        missing = REQUIRED_ARTIFACT_ROLES - records.keys()
        extra = records.keys() - REQUIRED_ARTIFACT_ROLES
        details = []
        if missing:
            details.append(f"missing roles: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unknown roles: {', '.join(sorted(extra))}")
        raise ValueError("incomplete source artifact registration (" + "; ".join(details) + ")")
    report_sha256 = _run_report_sha256(conn, run_id)
    for role, record in records.items():
        try:
            resolved_path = Path(record["resolved_path"]).resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"registered path for {role} no longer resolves") from exc
        if str(resolved_path) != record["resolved_path"]:
            raise ValueError(f"registered path for {role} no longer resolves to the recorded path")
        if not resolved_path.is_file():
            raise ValueError(f"registered artifact for {role} is no longer a file")
        try:
            payload = resolved_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"registered artifact for {role} cannot be read") from exc
        if len(payload) != record["byte_size"]:
            raise ValueError(f"registered artifact size changed for {role}")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"registered artifact SHA-256 changed for {role}")
    if records["swmm_rpt"]["sha256"] != report_sha256:
        raise ValueError("registered SWMM RPT hash does not match hydraulic quality manifest")
    return records
