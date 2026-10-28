"""Register and verify immutable links to the files behind a completed run."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "source_network_inp",
        "effective_inp",
        "event_source",
        "swmm_rpt",
        "swmm_out",
    }
)
_MANIFEST_FIELDS = ("artifact_role", "resolved_path", "sha256", "byte_size")


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _artifact_items(artifacts) -> list[tuple[object, object]]:
    if isinstance(artifacts, Mapping):
        return list(artifacts.items())
    if isinstance(artifacts, Iterable) and not isinstance(artifacts, (str, bytes)):
        try:
            return [tuple(item) for item in artifacts]
        except (TypeError, ValueError) as exc:
            raise ValueError("artifacts must contain role/path pairs") from exc
    raise ValueError("artifacts must be a role-to-path mapping or role/path pairs")


def _prepare_artifacts(artifacts) -> dict[str, dict]:
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
        except OSError as exc:
            raise ValueError(f"artifact for {role} cannot be read: {path}") from exc
        if not payload:
            raise ValueError(f"artifact for {role} is empty: {path}")
        prepared[role] = {
            "artifact_role": role,
            "resolved_path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        }

    missing = REQUIRED_ARTIFACT_ROLES - prepared.keys()
    if missing:
        raise ValueError(f"missing artifact role(s): {', '.join(sorted(missing))}")
    return prepared


def _run_identity(conn: sqlite3.Connection, run_id: int) -> dict:
    run = conn.execute(
        """SELECT run_id, network_id, scenario_id, config_sha256, status
           FROM runs WHERE run_id = ?""",
        (run_id,),
    ).fetchone()
    if run is None:
        raise ValueError(f"run_id does not exist: {run_id}")
    if run[4] != "COMPLETE":
        raise ValueError("source artifacts require a COMPLETE run")
    quality = conn.execute(
        "SELECT manifest_json FROM run_hydraulic_quality WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if quality is None:
        raise ValueError("source artifacts require immutable hydraulic quality")
    try:
        qa_manifest = json.loads(quality[0])
        report_sha256 = qa_manifest["flooding_report"]["report_sha256"]
    except (TypeError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError("hydraulic quality manifest has no flooding report hash") from exc
    if not _valid_sha256(report_sha256):
        raise ValueError("hydraulic quality manifest has an invalid flooding report hash")
    return {
        "run_id": run[0],
        "network_id": run[1],
        "scenario_id": run[2],
        "config_sha256": run[3],
        "report_sha256": report_sha256,
    }


def _verify_network_inp_against_database(
    conn: sqlite3.Connection, identity: dict, source_artifact: dict
) -> None:
    network = conn.execute(
        "SELECT network_sha256, inp_bytes FROM networks WHERE network_id = ?",
        (identity["network_id"],),
    ).fetchone()
    if network is None:
        raise ValueError("source network INP has no matching network record")
    stored_hash, inp_bytes = network[0], network[1]
    database_bytes_hash = hashlib.sha256(inp_bytes).hexdigest()
    if stored_hash != database_bytes_hash:
        raise ValueError("network INP database bytes do not match networks.network_sha256")
    if source_artifact["sha256"] != stored_hash:
        raise ValueError("source network INP file hash does not match the database network INP")


def _manifest_payload(identity: dict, records: dict[str, dict]) -> dict:
    artifacts = [
        {field: records[role][field] for field in _MANIFEST_FIELDS}
        for role in sorted(REQUIRED_ARTIFACT_ROLES)
    ]
    return {
        "run_id": identity["run_id"],
        "network_id": identity["network_id"],
        "scenario_id": identity["scenario_id"],
        "config_sha256": identity["config_sha256"],
        "artifacts": artifacts,
    }


def _manifest_digest(identity: dict, records: dict[str, dict]) -> str:
    return hashlib.sha256(_canonical_json(_manifest_payload(identity, records)).encode("utf-8")).hexdigest()


def _load_records(conn: sqlite3.Connection, run_id: int) -> dict[str, dict]:
    rows = conn.execute(
        """SELECT artifact_role, registered_network_id, registered_scenario_id,
                  resolved_path, sha256, byte_size, source_manifest_sha256,
                  registered_at_utc
           FROM run_source_artifacts WHERE run_id = ? ORDER BY artifact_role""",
        (run_id,),
    ).fetchall()
    return {
        row[0]: {
            "artifact_role": row[0],
            "registered_network_id": row[1],
            "registered_scenario_id": row[2],
            "resolved_path": row[3],
            "sha256": row[4],
            "byte_size": row[5],
            "source_manifest_sha256": row[6],
            "registered_at_utc": row[7],
        }
        for row in rows
    }


def _verify_registered_rows(conn: sqlite3.Connection, run_id: int) -> dict:
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

    identity = _run_identity(conn, run_id)
    for role, record in records.items():
        if (
            record["registered_network_id"] != identity["network_id"]
            or record["registered_scenario_id"] != identity["scenario_id"]
        ):
            raise ValueError(f"registered run identity snapshot mismatch for {role}")

    digest = _manifest_digest(identity, records)
    if any(record["source_manifest_sha256"] != digest for record in records.values()):
        raise ValueError("source manifest digest does not match the registered artifact records")
    _verify_network_inp_against_database(
        conn, identity, records["source_network_inp"]
    )
    if records["swmm_rpt"]["sha256"] != identity["report_sha256"]:
        raise ValueError("registered SWMM RPT hash does not match hydraulic quality manifest")
    return {
        "run_id": identity["run_id"],
        "network_id": identity["network_id"],
        "scenario_id": identity["scenario_id"],
        "config_sha256": identity["config_sha256"],
        "source_manifest_sha256": digest,
        "artifacts": records,
    }


def register_run_source_artifacts(
    conn: sqlite3.Connection, run_id: int, artifacts
) -> dict:
    """Atomically register five linked files and return their canonical manifest."""
    try:
        run_id = int(run_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("run_id must be an integer") from exc
    prepared = _prepare_artifacts(artifacts)
    identity = _run_identity(conn, run_id)
    _verify_network_inp_against_database(
        conn, identity, prepared["source_network_inp"]
    )
    if prepared["swmm_rpt"]["sha256"] != identity["report_sha256"]:
        raise ValueError("SWMM RPT hash does not match hydraulic quality manifest")
    if conn.execute(
        "SELECT 1 FROM run_source_artifacts WHERE run_id = ? LIMIT 1", (run_id,)
    ).fetchone():
        raise ValueError(f"source artifacts already registered for run_id {run_id}")

    digest = _manifest_digest(identity, prepared)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    conn.execute("SAVEPOINT register_run_source_artifacts")
    try:
        for role in sorted(REQUIRED_ARTIFACT_ROLES):
            record = prepared[role]
            conn.execute(
                """INSERT INTO run_source_artifacts (
                       run_id, artifact_role, registered_network_id,
                       registered_scenario_id, resolved_path, sha256,
                       byte_size, source_manifest_sha256, registered_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    role,
                    identity["network_id"],
                    identity["scenario_id"],
                    record["resolved_path"],
                    record["sha256"],
                    record["byte_size"],
                    digest,
                    timestamp,
                ),
            )
        manifest = _verify_registered_rows(conn, run_id)
        for role, record in manifest["artifacts"].items():
            resolved_path = Path(record["resolved_path"]).resolve(strict=True)
            if str(resolved_path) != record["resolved_path"]:
                raise ValueError(f"registered path for {role} changed during registration")
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
        conn.execute("RELEASE SAVEPOINT register_run_source_artifacts")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT register_run_source_artifacts")
        conn.execute("RELEASE SAVEPOINT register_run_source_artifacts")
        raise
    return manifest


def verify_run_source_artifacts(conn: sqlite3.Connection, run_id: int) -> dict:
    """Verify identity, manifest digest and current bytes for every linked file."""
    try:
        run_id = int(run_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("run_id must be an integer") from exc
    manifest = _verify_registered_rows(conn, run_id)
    for role, record in manifest["artifacts"].items():
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
    return manifest
