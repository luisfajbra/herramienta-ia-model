"""Create immutable, explicit run-quality decisions for training admission."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3


QUALITY_POLICY_VERSION = "hydraulic_quality_v1"
_DRAINAGE_STATES = {"drained", "not_drained", "unverified"}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_hydraulic_quality_decision(
    *,
    continuity_error_pct: float | None,
    max_continuity_error_pct: float,
    continuity_limit_rationale: str,
    finite_coverage_ok: bool,
    drainage_state: str,
    initial_condition: dict,
    flooding_report_manifest: dict | None,
) -> dict:
    """Build the deterministic software QA decision without database I/O.

    This function does not certify engineering suitability. It applies the
    explicit software gate: finite measured continuity within the supplied
    limit, complete finite output coverage, a drained state, known initial
    condition provenance, and a validated flooding-report provenance record.
    It can be used before database run IDs exist, but callers must only pass a
    report manifest after parsing labels successfully.
    """
    try:
        limit = float(max_continuity_error_pct)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_continuity_error_pct must be finite and non-negative") from exc
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("max_continuity_error_pct must be finite and non-negative")
    if not isinstance(continuity_limit_rationale, str) or not continuity_limit_rationale.strip():
        raise ValueError("continuity_limit_rationale must be a non-empty string")
    if type(finite_coverage_ok) is not bool:
        raise ValueError("finite_coverage_ok must be a boolean")
    if drainage_state not in _DRAINAGE_STATES:
        raise ValueError(f"drainage_state must be one of {sorted(_DRAINAGE_STATES)}")
    if not isinstance(initial_condition, dict):
        raise ValueError("initial_condition must be an object")

    if continuity_error_pct is None:
        measured_continuity = None
    else:
        try:
            measured_continuity = float(continuity_error_pct)
        except (TypeError, ValueError):
            measured_continuity = None
    if measured_continuity is not None and not math.isfinite(measured_continuity):
        measured_continuity = None
    reasons = []
    if measured_continuity is None:
        reasons.append("continuity_error_missing")
    elif abs(measured_continuity) > limit:
        reasons.append("continuity_limit_exceeded")
    if not finite_coverage_ok:
        reasons.append("finite_coverage_failed")
    if drainage_state != "drained":
        reasons.append(f"drainage_{drainage_state}")
    if not initial_condition.get("policy_id"):
        reasons.append("initial_condition_provenance_missing")
    elif not _valid_sha256(initial_condition.get("state_sha256")):
        reasons.append("initial_condition_provenance_invalid")
    if (
        isinstance(flooding_report_manifest, dict)
        and flooding_report_manifest.get("failure_code")
    ):
        reasons.append("flooding_report_invalid")
    elif not isinstance(flooding_report_manifest, dict) or not all(
        (
            flooding_report_manifest.get("parser"),
            flooding_report_manifest.get("parser_version"),
            _valid_sha256(flooding_report_manifest.get("report_sha256")),
        )
    ):
        reasons.append("flooding_report_provenance_missing")

    reason_codes = sorted(set(reasons))
    eligible = not reason_codes
    policy = {
        "policy_version": QUALITY_POLICY_VERSION,
        "max_continuity_error_pct": limit,
        "continuity_limit_rationale": continuity_limit_rationale.strip(),
        "continuity_comparison": "absolute_value",
        "require_finite_coverage": True,
        "required_drainage_state": "drained",
        "require_initial_condition_provenance": True,
        "require_flooding_report_provenance": True,
    }
    policy_json = _canonical_json(policy)
    manifest = {
        "policy": policy,
        "continuity_error_pct": measured_continuity,
        "finite_coverage_ok": finite_coverage_ok,
        "drainage_state": drainage_state,
        "initial_condition": initial_condition,
        "flooding_report": flooding_report_manifest,
        "eligible": eligible,
        "reason_codes": reason_codes,
    }
    policy_sha256 = _sha256_text(policy_json)
    return {
        **manifest,
        "policy_sha256": policy_sha256,
        "initial_condition_json": _canonical_json(initial_condition),
        "reason_codes_json": _canonical_json(reason_codes),
        "manifest_json": _canonical_json(manifest),
    }


def record_run_hydraulic_quality(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    continuity_error_pct: float | None,
    max_continuity_error_pct: float,
    continuity_limit_rationale: str,
    finite_coverage_ok: bool,
    drainage_state: str,
    initial_condition: dict,
    flooding_report_manifest: dict | None,
) -> dict:
    """Persist one append-only QA assessment after verifying its source run."""
    decision = build_hydraulic_quality_decision(
        continuity_error_pct=continuity_error_pct,
        max_continuity_error_pct=max_continuity_error_pct,
        continuity_limit_rationale=continuity_limit_rationale,
        finite_coverage_ok=finite_coverage_ok,
        drainage_state=drainage_state,
        initial_condition=initial_condition,
        flooding_report_manifest=flooding_report_manifest,
    )
    run = conn.execute(
        "SELECT status, continuity_error_pct FROM runs WHERE run_id = ?",
        (int(run_id),),
    ).fetchone()
    if run is None:
        raise ValueError(f"run_id does not exist: {run_id}")
    if run[0] != "COMPLETE":
        raise ValueError("hydraulic quality can only be recorded for COMPLETE runs")
    measured_continuity = decision["continuity_error_pct"]
    if run[1] != measured_continuity:
        raise ValueError("continuity_error_pct does not match the stored run value")

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    conn.execute(
        """INSERT INTO run_hydraulic_quality (
               run_id, policy_version, policy_json, policy_sha256,
               continuity_error_pct, finite_coverage_ok, drainage_state,
               initial_condition_json, eligible, reason_codes_json,
               manifest_json, created_at_utc
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            int(run_id),
            QUALITY_POLICY_VERSION,
            _canonical_json(decision["policy"]),
            decision["policy_sha256"],
            measured_continuity,
            int(decision["finite_coverage_ok"]),
            decision["drainage_state"],
            decision["initial_condition_json"],
            int(decision["eligible"]),
            decision["reason_codes_json"],
            decision["manifest_json"],
            created_at,
        ),
    )
    return {**decision, "created_at_utc": created_at}
