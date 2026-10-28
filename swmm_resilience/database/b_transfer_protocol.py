"""Freeze a B simulation protocol and anchor it in a separate append-only DB."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from ..simulation.initial_conditions import build_initial_condition_manifest


_POLICY_FIELDS = frozenset({
    "max_continuity_error_pct", "continuity_limit_rationale", "drain_down_hours",
    "drain_down_extension_hours", "max_drain_retries",
    "tail_window_hours", "min_report_steps", "inflow_tolerance_lps",
    "flooding_tolerance_lps", "storage_change_tolerance_m3", "residual_flow_tolerance_lps",
})
_SCENARIO_FIELDS = frozenset({
    "scenario_key", "shape_id", "factor_mult", "duracion_horas", "tiempo_al_pico_h",
})


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_json(value: object) -> str:
    return _sha_bytes(_canonical(value).encode("utf-8"))


def _check_policy(policy: object) -> dict:
    if not isinstance(policy, dict) or set(policy) != _POLICY_FIELDS:
        raise ValueError("hydraulic_policy must declare the complete frozen QA policy")
    for name in _POLICY_FIELDS - {"continuity_limit_rationale"}:
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"hydraulic_policy.{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"hydraulic_policy.{name} must be finite")
        if name == "min_report_steps":
            if int(value) != value or value < 2:
                raise ValueError("hydraulic_policy.min_report_steps must be an integer >= 2")
        elif name == "max_drain_retries":
            if int(value) != value or value < 0:
                raise ValueError("hydraulic_policy.max_drain_retries must be an integer >= 0")
        elif value < 0 or (name in {"drain_down_hours", "tail_window_hours"} and value <= 0):
            raise ValueError(f"hydraulic_policy.{name} is outside its valid range")
    if not isinstance(policy["continuity_limit_rationale"], str) or not policy["continuity_limit_rationale"].strip():
        raise ValueError("hydraulic_policy.continuity_limit_rationale must be non-empty")
    if policy["tail_window_hours"] > policy["drain_down_hours"]:
        raise ValueError("tail_window_hours cannot exceed drain_down_hours")
    return json.loads(_canonical(policy))


def _anchor_payload(row: tuple) -> dict:
    return {
        "protocol_path": row[1], "protocol_sha256": row[2], "git_commit": row[3],
        "execution_context": json.loads(row[4]), "previous_anchor_sha256": row[5],
        "created_at_utc": row[6],
    }


def _initialize_anchor_db(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS protocol_anchors (
        anchor_id INTEGER PRIMARY KEY AUTOINCREMENT,
        protocol_path TEXT NOT NULL UNIQUE,
        protocol_sha256 TEXT NOT NULL CHECK(length(protocol_sha256)=64),
        git_commit TEXT NOT NULL,
        execution_context_json TEXT NOT NULL CHECK(json_valid(execution_context_json)),
        previous_anchor_sha256 TEXT,
        created_at_utc TEXT NOT NULL,
        anchor_sha256 TEXT NOT NULL CHECK(length(anchor_sha256)=64)
    )""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS protocol_anchors_no_update
        BEFORE UPDATE ON protocol_anchors BEGIN
            SELECT RAISE(ABORT, 'protocol anchor is append-only');
        END""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS protocol_anchors_no_delete
        BEFORE DELETE ON protocol_anchors BEGIN
            SELECT RAISE(ABORT, 'protocol anchor is append-only');
        END""")


def _verify_anchor_chain(conn: sqlite3.Connection) -> list[tuple]:
    rows = conn.execute("""SELECT anchor_id,protocol_path,protocol_sha256,git_commit,
        execution_context_json,previous_anchor_sha256,created_at_utc,anchor_sha256
        FROM protocol_anchors ORDER BY anchor_id""").fetchall()
    previous = None
    for index, row in enumerate(rows, start=1):
        payload = _anchor_payload(row)
        if row[0] != index or row[5] != previous or row[7] != _sha_json(payload):
            raise ValueError("external protocol anchor chain is invalid or was modified")
        previous = row[7]
    return rows


def freeze_b_transfer_protocol(
    experiment_root: str | Path,
    protocol: dict,
    *,
    anchor_db_path: str | Path,
    git_commit: str,
    execution_context: dict,
) -> dict:
    """Write an immutable pre-run protocol and append its receipt externally.

    The anchor DB must live outside ``experiment_root``. It provides separation
    and append-only guards, not cryptographic authenticity against an operator
    able to replace that external DB.
    """
    root = Path(experiment_root).resolve(strict=True)
    anchor_path = Path(anchor_db_path).resolve()
    protocol_path = root / "protocol.json"
    if protocol_path.exists() or protocol_path.is_symlink():
        raise FileExistsError("protocol.json already exists; frozen protocols cannot be replaced")
    try:
        anchor_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("protocol anchor DB must be outside the experiment root")
    if not isinstance(protocol, dict) or set(protocol) != {
        "schema_version", "status", "network", "expected_scenario_keys", "scenarios", "hydraulic_policy",
    }:
        raise ValueError("protocol input has missing or unsupported fields")
    if protocol["schema_version"] != "b_transfer_protocol_v2" or protocol["status"] != "frozen_before_run":
        raise ValueError("protocol must use b_transfer_protocol_v2/frozen_before_run")
    policy = _check_policy(protocol["hydraulic_policy"])
    keys = protocol["expected_scenario_keys"]
    definitions = protocol["scenarios"]
    if (not isinstance(keys, list) or not keys or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys) or not isinstance(definitions, list) or len(definitions) != len(keys)):
        raise ValueError("protocol requires an ordered unique scenario-key list and matching definitions")
    if any(not isinstance(item, dict) or set(item) != _SCENARIO_FIELDS for item in definitions):
        raise ValueError("every protocol scenario must declare exact identity and timing fields")
    if [item["scenario_key"] for item in definitions] != keys:
        raise ValueError("scenario definitions must follow expected_scenario_keys order")
    network = protocol["network"]
    if not isinstance(network, dict) or set(network) - {"path", "sha256", "name"} or "path" not in network:
        raise ValueError("network requires a relative path and optional declared hash/name")
    if not isinstance(network["path"], str) or not network["path"].strip():
        raise ValueError("network path must be explicit")
    network_is_absolute = Path(network["path"]).is_absolute()
    if network_is_absolute and not network.get("sha256"):
        raise ValueError("an absolute source network link requires an explicit SHA-256")
    network_path = (Path(network["path"]) if network_is_absolute else (root / network["path"])).resolve(strict=True)
    if not network_is_absolute:
        network_path.relative_to(root)
    network_sha = _sha_bytes(network_path.read_bytes())
    if network.get("sha256") not in (None, network_sha):
        raise ValueError("declared network SHA-256 does not match source INP")
    initial_condition = build_initial_condition_manifest(network_path)
    frozen = {
        **protocol,
        "network": {key: value for key, value in {
            "path": str(network_path) if network_is_absolute else Path(network["path"]).as_posix(), "sha256": network_sha,
            "name": network.get("name"),
        }.items() if value is not None},
        "hydraulic_policy": policy,
        "initial_condition": initial_condition,
    }
    if not isinstance(execution_context, dict) or not execution_context:
        raise ValueError("execution_context must be an explicit non-empty object")
    context_json = _canonical(execution_context)
    if not isinstance(git_commit, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git_commit) is None:
        raise ValueError("git_commit must be a full lowercase Git object ID")
    protocol_bytes = (json.dumps(frozen, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    created_protocol = False
    try:
        with sqlite3.connect(anchor_path) as conn:
            _initialize_anchor_db(conn)
            conn.commit()
            rows = _verify_anchor_chain(conn)
            if any(Path(row[1]).resolve() == protocol_path for row in rows):
                raise FileExistsError("this protocol path already has an external anchor")
        with protocol_path.open("xb") as stream:
            stream.write(protocol_bytes)
        created_protocol = True
        with sqlite3.connect(anchor_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _initialize_anchor_db(conn)
            rows = _verify_anchor_chain(conn)
            previous = rows[-1][7] if rows else None
            timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            payload = {
                "protocol_path": str(protocol_path), "protocol_sha256": _sha_bytes(protocol_bytes),
                "git_commit": git_commit, "execution_context": json.loads(context_json),
                "previous_anchor_sha256": previous, "created_at_utc": timestamp,
            }
            anchor_sha = _sha_json(payload)
            cur = conn.execute("""INSERT INTO protocol_anchors
                (protocol_path,protocol_sha256,git_commit,execution_context_json,
                 previous_anchor_sha256,created_at_utc,anchor_sha256)
                VALUES (?,?,?,?,?,?,?)""", (
                payload["protocol_path"], payload["protocol_sha256"], git_commit, context_json,
                previous, timestamp, anchor_sha,
            ))
            anchor_id = int(cur.lastrowid)
            conn.commit()
        return {"anchor_id": anchor_id, "anchor_sha256": anchor_sha,
                "protocol_sha256": payload["protocol_sha256"], "git_commit": git_commit,
                "execution_context": json.loads(context_json)}
    except Exception:
        if created_protocol:
            protocol_path.unlink(missing_ok=True)
        raise


def verify_b_transfer_protocol_anchor(
    anchor_db_path: str | Path,
    protocol_path: str | Path,
    receipt: dict,
) -> dict:
    """Verify a frozen protocol against its independent append-only receipt."""
    if not isinstance(receipt, dict) or set(receipt) != {
        "anchor_id", "anchor_sha256", "protocol_sha256", "git_commit", "execution_context",
    }:
        raise ValueError("protocol anchor receipt has missing or unsupported fields")
    path = Path(protocol_path).resolve(strict=True)
    protocol_sha = _sha_bytes(path.read_bytes())
    with sqlite3.connect(f"file:{Path(anchor_db_path).resolve()}?mode=ro", uri=True) as conn:
        rows = _verify_anchor_chain(conn)
    matches = [row for row in rows if Path(row[1]).resolve() == path]
    if len(matches) != 1:
        raise ValueError("protocol has no unique external pre-run anchor")
    row = matches[0]
    if (row[0] != receipt.get("anchor_id") or row[2] != protocol_sha
            or row[7] != receipt.get("anchor_sha256")
            or row[3] != receipt.get("git_commit")
            or json.loads(row[4]) != receipt.get("execution_context")
            or receipt.get("protocol_sha256") != protocol_sha):
        raise ValueError("protocol bytes or external anchor receipt do not match the pre-run seal")
    return {"anchor_id": row[0], "anchor_sha256": row[7], "protocol_sha256": protocol_sha,
            "git_commit": row[3], "execution_context": json.loads(row[4])}
