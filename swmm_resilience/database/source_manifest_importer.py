"""Import explicitly frozen simulation manifests into immutable run links.

The importer deliberately accepts no filename inference: every artifact path
must be present in a frozen protocol or batch record and hash to its declared
SHA-256 before a database run can be linked.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3

from .run_source_artifacts import register_run_source_artifacts


_RESULT_ROLES = ("effective_inp", "event_source", "swmm_rpt", "swmm_out")
_OUTPUT_NAME = "experiment_source_manifest.json"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _read_json(path: Path, label: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return data


def _resolve_recorded_file(root: Path, relative_path: object, expected_hash: object, label: str) -> dict:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{label} requires an explicit relative path")
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"{label} path must be relative to the recorded experiment root")
    try:
        resolved = (root / path).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} path is missing or escapes the experiment root") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    payload = resolved.read_bytes()
    if not payload:
        raise ValueError(f"{label} is empty")
    digest = hashlib.sha256(payload).hexdigest()
    declared = _sha256(expected_hash, f"{label} sha256")
    if digest != declared:
        raise ValueError(f"{label} SHA-256 mismatch")
    return {
        "relative_path": resolved.relative_to(root).as_posix(),
        "resolved_path": str(resolved),
        "sha256": digest,
        "byte_size": len(payload),
    }


def _scenario_key(conn: sqlite3.Connection, network_sha256: str, shape_id: str, factor: float) -> tuple[int, dict]:
    rows = conn.execute(
        """SELECT r.run_id, r.status, r.network_id, r.scenario_id,
                  s.scenario_key, s.shape_id, s.factor_mult, q.eligible,
                  q.manifest_json
           FROM networks n
           JOIN runs r ON r.network_id = n.network_id
           JOIN scenarios s ON s.scenario_id = r.scenario_id
                            AND s.network_id = r.network_id
           LEFT JOIN run_hydraulic_quality q ON q.run_id = r.run_id
           WHERE n.network_sha256 = ? AND s.shape_id = ? AND s.factor_mult = ?
           ORDER BY r.run_id""",
        (network_sha256, shape_id, factor),
    ).fetchall()
    if not rows:
        raise ValueError(
            f"no database run for exact key (network hash, shape_id={shape_id!r}, factor_mult={factor})"
        )
    if len(rows) != 1:
        raise ValueError(
            f"ambiguous database run mapping for exact key (network hash, shape_id={shape_id!r}, factor_mult={factor})"
        )
    row = rows[0]
    if row[1] != "COMPLETE" or row[7] != 1:
        raise ValueError(f"database run_id {row[0]} is ineligible")
    try:
        qa = json.loads(row[8])
        qa_rpt_hash = qa["flooding_report"]["report_sha256"]
        if qa.get("eligible") is not True or qa.get("reason_codes") != []:
            raise ValueError
    except (TypeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise ValueError(f"database run_id {row[0]} has an invalid or ineligible QA manifest") from exc
    return row[0], {
        "network_id": row[2], "scenario_id": row[3], "scenario_key": row[4],
        "shape_id": row[5], "factor_mult": row[6], "report_sha256": _sha256(qa_rpt_hash, "QA RPT hash"),
    }


def _prepare_import(conn: sqlite3.Connection, root: Path) -> tuple[dict, list[dict]]:
    protocol_path = root / "protocol.json"
    batch_path = root / "batch_progress.json"
    protocol = _read_json(protocol_path, "protocol.json")
    batch = _read_json(batch_path, "batch_progress.json")
    protocol_bytes_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if _sha256(batch.get("protocol_sha256"), "batch protocol_sha256") != protocol_bytes_hash:
        raise ValueError("batch_progress protocol seal does not match protocol.json bytes")
    if not isinstance(protocol.get("status"), str) or not protocol["status"].startswith("frozen_"):
        raise ValueError("protocol status must identify a frozen protocol")
    if batch.get("status") != "complete":
        raise ValueError("batch_progress must have complete status")
    raw_results = batch.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("batch_progress results must be a non-empty list")
    expected = protocol.get("expected_scenarios")
    if type(expected) is not int or expected != len(raw_results):
        raise ValueError("batch result count does not match frozen protocol expected_scenarios")

    network = _resolve_recorded_file(
        root, protocol.get("input_inp"), protocol.get("input_inp_sha256"), "source network INP"
    )
    # Shape source paths and hashes, when frozen in protocol.json, are checked
    # independently from each event series used by a run.
    frozen_shapes = protocol.get("shape_csvs")
    if not isinstance(frozen_shapes, dict):
        raise ValueError("protocol shape_csvs must be an object")
    checked_shapes = {}
    prepared = []
    seen_keys = set()
    seen_run_ids = set()
    for index, record in enumerate(raw_results):
        if not isinstance(record, dict):
            raise ValueError(f"batch result {index} must be an object")
        shape_id = record.get("shape_id")
        if not isinstance(shape_id, str) or not shape_id.strip():
            raise ValueError(f"batch result {index} has no exact shape_id")
        factor_raw = record.get("factor_mult", record.get("factor"))
        if isinstance(factor_raw, bool):
            raise ValueError(f"batch result {index} has invalid factor_mult")
        try:
            factor = float(factor_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"batch result {index} has invalid factor_mult") from exc
        if not math.isfinite(factor):
            raise ValueError(f"batch result {index} has non-finite factor_mult")
        key = (shape_id, factor)
        if key in seen_keys:
            raise ValueError(f"duplicate batch record for shape_id={shape_id!r}, factor_mult={factor}")
        seen_keys.add(key)
        if shape_id != "base" and shape_id not in frozen_shapes:
            raise ValueError(f"shape_id {shape_id!r} is absent from frozen protocol shape_csvs")
        if shape_id in frozen_shapes and shape_id not in checked_shapes:
            shape = frozen_shapes[shape_id]
            if not isinstance(shape, dict):
                raise ValueError(f"frozen shape source {shape_id!r} must be an object")
            checked_shapes[shape_id] = _resolve_recorded_file(
                root, shape.get("path"), shape.get("sha256"), f"frozen shape source {shape_id}"
            )

        if record.get("eligible_by_software_qa") is not True or record.get("reason_codes") != []:
            raise ValueError(f"batch record {index} is ineligible by software QA")
        scenario_key = record.get("scenario_key")
        if not isinstance(scenario_key, str) or not scenario_key:
            raise ValueError(f"batch record {index} requires the exact scenario_key")
        run_id, identity = _scenario_key(conn, network["sha256"], shape_id, factor)
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate database run mapping to run_id {run_id}")
        seen_run_ids.add(run_id)
        if identity["scenario_key"] != scenario_key:
            raise ValueError(f"scenario_key mismatch for run_id {run_id}")

        result_artifacts = record.get("artifacts")
        if not isinstance(result_artifacts, dict) or set(result_artifacts) != set(_RESULT_ROLES):
            raise ValueError(f"batch record {index} must explicitly reference all four result artifact roles")
        role_records = {"source_network_inp": network}
        for role in _RESULT_ROLES:
            artifact = result_artifacts[role]
            if not isinstance(artifact, dict):
                raise ValueError(f"batch result {index} {role} reference must be an object")
            role_records[role] = _resolve_recorded_file(
                root, artifact.get("path"), artifact.get("sha256"), f"batch result {index} {role}"
            )
        if record.get("event_series_sha256") != role_records["event_source"]["sha256"]:
            raise ValueError(f"event-series hash mismatch for batch result {index}")
        if record.get("rpt") != result_artifacts["swmm_rpt"].get("path") or record.get("rpt_sha256") != role_records["swmm_rpt"]["sha256"]:
            raise ValueError(f"legacy RPT reference mismatch for batch result {index}")
        flooding_report = record.get("flooding_report")
        if not isinstance(flooding_report, dict) or flooding_report.get("report_sha256") != role_records["swmm_rpt"]["sha256"]:
            raise ValueError(f"batch flooding-report hash mismatch for batch result {index}")
        if role_records["swmm_rpt"]["sha256"] != identity["report_sha256"]:
            raise ValueError(f"database QA RPT hash mismatch for run_id {run_id}")
        prepared.append({
            "run_id": run_id,
            "identity": {**identity, "scenario_key": scenario_key},
            "artifacts": role_records,
        })
    return {
        "protocol_sha256": protocol_bytes_hash,
        "network_sha256": network["sha256"],
        "checked_shape_sources": checked_shapes,
    }, prepared


def import_frozen_experiment_manifest(
    conn: sqlite3.Connection,
    experiment_root: str | Path,
    *,
    manifest_name: str = _OUTPUT_NAME,
) -> dict:
    """Verify a frozen protocol/batch pair, register exact run links and seal them.

    Protocol and batch input paths are relative to ``experiment_root``. The
    new manifest is created exclusively; existing manifests are never replaced.
    This function does not run SWMM or write scenario/feature/label data.
    """
    root = Path(experiment_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("experiment_root must be an existing directory")
    if Path(manifest_name).name != manifest_name or manifest_name in {".", ".."}:
        raise ValueError("manifest_name must be a simple filename")
    manifest_path = root / manifest_name
    if manifest_path.exists():
        raise FileExistsError(f"immutable experiment manifest already exists: {manifest_path}")
    source, prepared = _prepare_import(conn, root)

    conn.execute("SAVEPOINT import_frozen_experiment_manifest")
    wrote_manifest = False
    try:
        imported_runs = []
        links = []
        for item in prepared:
            db_links = {role: artifact["resolved_path"] for role, artifact in item["artifacts"].items()}
            verified = register_run_source_artifacts(conn, item["run_id"], db_links)
            artifacts = {}
            for role, record in verified["artifacts"].items():
                relative = Path(record["resolved_path"]).resolve(strict=True).relative_to(root).as_posix()
                artifacts[role] = {
                    "path": relative, "sha256": record["sha256"], "byte_size": record["byte_size"]
                }
                links.append({
                    "run_id": item["run_id"], "artifact_role": role,
                    "path": relative, "sha256": record["sha256"],
                    "byte_size": record["byte_size"],
                    "source_manifest_sha256": verified["source_manifest_sha256"],
                })
            imported_runs.append({
                "run_id": item["run_id"],
                "network_id": item["identity"]["network_id"],
                "scenario_id": item["identity"]["scenario_id"],
                "scenario_key": item["identity"]["scenario_key"],
                "shape_id": item["identity"]["shape_id"],
                "factor_mult": item["identity"]["factor_mult"],
                "source_manifest_sha256": verified["source_manifest_sha256"],
                "artifacts": artifacts,
            })
        link_hash = hashlib.sha256(_canonical_json(links).encode("utf-8")).hexdigest()
        payload = {
            "schema_version": "frozen_experiment_source_manifest_v1",
            "experiment_root": str(root),
            "protocol_path": "protocol.json",
            "batch_progress_path": "batch_progress.json",
            "protocol_id": _read_json(root / "protocol.json", "protocol.json").get("protocol_id"),
            "protocol_sha256": source["protocol_sha256"],
            "batch_progress_sha256": hashlib.sha256((root / "batch_progress.json").read_bytes()).hexdigest(),
            "network_sha256": source["network_sha256"],
            "checked_shape_sources": {
                shape_id: {
                    "path": record["relative_path"], "sha256": record["sha256"],
                    "byte_size": record["byte_size"],
                }
                for shape_id, record in source["checked_shape_sources"].items()
            },
            "link_table_sha256": link_hash,
            "runs": imported_runs,
        }
        payload["manifest_sha256"] = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        stream = manifest_path.open("x", encoding="utf-8", newline="\n")
        # Mark ownership as soon as exclusive creation succeeds, so a failed
        # write/flush also removes this partial manifest during rollback.
        wrote_manifest = True
        with stream:
            stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        conn.execute("RELEASE SAVEPOINT import_frozen_experiment_manifest")
        return {**payload, "manifest_path": manifest_name}
    except Exception:
        if wrote_manifest:
            manifest_path.unlink(missing_ok=True)
        conn.execute("ROLLBACK TO SAVEPOINT import_frozen_experiment_manifest")
        conn.execute("RELEASE SAVEPOINT import_frozen_experiment_manifest")
        raise
