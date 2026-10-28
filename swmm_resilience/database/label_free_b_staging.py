"""Create label-free, source-verified B run rows before transfer prediction.

This staging protocol is separate from the historical-run importer: it starts
from a complete frozen manifest and allocates fresh run IDs in a new database.
It deliberately has no label argument or target-table write path.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from ..extraction.dynamic_features import (
    compute_scenario_dynamic_features,
    compute_tributary_dynamic_features,
)
from ..extraction.static_features import extract_static_features
from ..extraction.topology import build_network_graph, compute_topology_features
from ..ml.contracts import FEATURE_COLUMNS_V17, FEATURE_COLUMNS_V21, resolve_feature_contract
from ..ml.bench.schemas import KEY_COLUMNS
from ..simulation.swmm_api_io import load_inp
from ..simulation.swmm_api_io import read_continuity_error
from ..simulation.initial_conditions import build_initial_condition_manifest
from ..extraction.labels import FloodingReportError, read_flooding_volumes_with_manifest
from ..validation.drainage_state import classify_drainage_state, read_tail_window_observables
from ..validation.hydrograph_csv import load_scenario
from .b_transfer_protocol import verify_b_transfer_protocol_anchor
from .hydraulic_quality import build_hydraulic_quality_decision, record_run_hydraulic_quality
from .run_source_artifacts import register_run_source_artifacts, verify_run_source_artifacts


_ARTIFACT_ROLES = frozenset({"source_network_inp", "effective_inp", "event_source", "swmm_rpt", "swmm_out"})
_QA_FIELDS = frozenset({
    "continuity_error_pct", "max_continuity_error_pct", "continuity_limit_rationale",
    "finite_coverage_ok", "drainage_state", "initial_condition", "flooding_report",
})
_RUN_FIELDS = frozenset({"scenario_key", "hydraulic_quality", "artifacts"})
_PROTOCOL_SCENARIO_FIELDS = frozenset({
    "scenario_key", "shape_id", "factor_mult", "duracion_horas", "tiempo_al_pico_h",
})
_TARGET_NAMES = frozenset({
    "labels", "targets", "inunda", "vol_inundacion_m3", "peak_flooding_lps",
    "flooding_duration_min", "max_depth_m", "max_depth_ratio", "node_results",
})


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _reject_target_fields(value: object, where: str = "manifest") -> None:
    if isinstance(value, dict):
        forbidden = _TARGET_NAMES.intersection(value)
        if forbidden:
            raise ValueError(f"target/label fields are forbidden in {where}: {', '.join(sorted(forbidden))}")
        for key, item in value.items():
            _reject_target_fields(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_target_fields(item, f"{where}[{index}]")


def _artifact(root: Path, record: object, label: str, *, allow_external: bool = False) -> dict:
    allowed_fields = {"path", "sha256", "name"} if label == "source network INP" else {"path", "sha256"}
    if (not isinstance(record, dict) or not {"path", "sha256"}.issubset(record)
            or set(record) - allowed_fields):
        raise ValueError(f"{label} requires an explicit path and SHA-256")
    relative = record["path"]
    digest = record["sha256"]
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{label} path must be explicit")
    is_absolute = Path(relative).is_absolute()
    if is_absolute and not allow_external:
        raise ValueError(f"{label} path must remain inside the experiment root")
    if is_absolute and label == "source network INP" and digest is None:
        raise ValueError("an absolute source network path requires an explicit SHA-256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label} SHA-256 is invalid")
    try:
        path = (Path(relative) if is_absolute else (root / relative)).resolve(strict=True)
        if not is_absolute:
            path.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} path is missing or escapes the frozen root") from exc
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file")
    payload = path.read_bytes()
    actual = _hash_bytes(payload)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 mismatch")
    return {"path": path, "sha256": actual, "byte_size": len(payload)}


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite numeric") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return number


def _make_features(inp_path: Path, event_path: Path, feature_contract_id: str) -> tuple[pd.DataFrame, list[str], dict]:
    contract = resolve_feature_contract(feature_contract_id)
    if contract.contract_id not in {"tabular_v3_17", "tabular_v4_21"}:
        raise ValueError(f"unsupported transfer feature contract: {contract.contract_id}")
    static = extract_static_features(inp_path)
    if static.empty or static["node_id"].astype(str).duplicated().any():
        raise ValueError("effective INP must produce unique junction feature rows")
    static["node_id"] = static["node_id"].astype(str)
    full = compute_topology_features(static, inp_path)
    inp = load_inp(inp_path)
    graph, _ = build_network_graph(inp)
    expected_inflow_nodes = set(
        full.loc[full["base_inflow_lps"].astype(float) > 0.0, "node_id"].astype(str)
    )
    scenario = load_scenario(event_path, expected_nodes=expected_inflow_nodes)
    if not expected_inflow_nodes:
        raise ValueError("effective INP has no positive inflow nodes for the frozen event")
    peak_map = {node: max(value for _, value in series) for node, series in scenario.node_series.items()}
    representative = next(iter(scenario.node_series.values()))
    peak_time = max(representative, key=lambda point: point[1])[0]
    dynamic = compute_scenario_dynamic_features(
        full, peak_map, graph, duracion_horas=scenario.last_time_hours,
        tiempo_al_pico_h=peak_time,
    )
    if contract.contract_id == "tabular_v4_21":
        capacity = dict(zip(full["node_id"].astype(str), full["upstream_capacity_lps"]))
        tributary = compute_tributary_dynamic_features(
            full["node_id"].astype(str).tolist(), scenario.node_series, graph, capacity
        )
        dynamic = dynamic.merge(tributary, on="node_id", how="left", validate="one_to_one")
    base = full.drop(columns=[
        column for column in (
            "factor_mult", "q_pico_nodo", "q_pico_acum_escalado", "duracion_horas", "tiempo_al_pico_h",
        ) if column in full.columns
    ])
    merged = base.merge(dynamic, on="node_id", how="left", validate="one_to_one")
    columns = FEATURE_COLUMNS_V17 if contract.contract_id == "tabular_v3_17" else FEATURE_COLUMNS_V21
    features = contract.validate_frame(merged.loc[:, list(columns)]).reset_index(drop=True)
    node_ids = merged["node_id"].astype(str).tolist()
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate node features from the verified effective INP")
    return features, node_ids, {
        "node_series": scenario.node_series, "full": full, "contract": contract,
        "duration_h": scenario.last_time_hours, "peak_time_h": peak_time,
    }


def _inp_object_ids(inp_path: Path | str) -> tuple[set[str], set[str]]:
    """Return the complete hydraulic node/link identity sets from one parsed INP."""
    inp = load_inp(inp_path)
    node_ids = {
        str(object_id)
        for section in ("JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS")
        if section in inp
        for object_id in inp[section]
    }
    link_ids = {
        str(object_id)
        for section in ("CONDUITS", "PUMPS", "ORIFICES", "WEIRS", "OUTLETS")
        if section in inp
        for object_id in inp[section]
    }
    return node_ids, link_ids


def _derive_hydraulic_quality(
    *, source_inp: Path, effective_inp: Path, rpt_path: Path, out_path: Path,
    expected_node_ids: set[str], expected_link_ids: set[str], policy: dict,
) -> dict:
    """Recalculate run QA from the verified hydraulic artifacts, never caller values."""
    try:
        continuity = read_continuity_error(rpt_path)
    except Exception:
        continuity = None
    try:
        volumes, report_manifest = read_flooding_volumes_with_manifest(rpt_path)
        report_ids = set(volumes["node_id"].astype(str))
        # The shared parser validates the complete summary, row uniqueness,
        # finite/non-negative values, and the explicit no-flood marker. Zero
        # labels for absent nodes are implicit; no target vector is materialized.
        finite_labels = report_ids.issubset(expected_node_ids)
        label_diagnostic = {"reported_node_ids_known": finite_labels, "reported_row_count": len(volumes)}
    except (FloodingReportError, OSError, ValueError, TypeError, KeyError) as exc:
        try:
            report_hash = _hash_bytes(rpt_path.read_bytes())
        except OSError:
            report_hash = None
        report_manifest = {
            "schema_version": "flooding_report_v1", "parser": "failed_closed",
            "parser_version": "1", "report_sha256": report_hash,
            "failure_code": getattr(exc, "code", "label_parse_failed"),
        }
        finite_labels = False
        label_diagnostic = {"parse_failed": getattr(exc, "code", type(exc).__name__)}
    try:
        observables = read_tail_window_observables(
            out_path, drain_down_hours=policy["drain_down_hours"],
            tail_window_hours=policy["tail_window_hours"],
        )
        drainage_state, drainage_manifest = classify_drainage_state(
            observables,
            inflow_tolerance_lps=policy["inflow_tolerance_lps"],
            flooding_tolerance_lps=policy["flooding_tolerance_lps"],
            storage_change_tolerance_m3=policy["storage_change_tolerance_m3"],
            residual_flow_tolerance_lps=policy["residual_flow_tolerance_lps"],
            window_hours=min(policy["tail_window_hours"], policy["drain_down_hours"]),
            min_report_steps=policy["min_report_steps"],
        )
        required_nodes = ("node_total_inflow_lps", "node_flooding_lps", "node_volume_m3")
        node_coverage = all(
            isinstance(observables.get(name), dict)
            and set(observables[name]) == expected_node_ids
            for name in required_nodes
        )
        link_coverage = (
            isinstance(observables.get("link_flow_lps"), dict)
            and set(observables["link_flow_lps"]) == expected_link_ids
            and bool(observables.get("has_link_data"))
        )
        step_count = observables.get("n_report_steps")
        finite_out = (node_coverage and link_coverage and isinstance(step_count, int)
                      and step_count >= policy["min_report_steps"])
        out_diagnostic = {"node_coverage": node_coverage, "n_report_steps": step_count,
                          "expected_node_count": len(expected_node_ids),
                          "observed_node_ids": {name: sorted(observables[name]) if isinstance(observables.get(name), dict) else []
                                                 for name in required_nodes},
                          "link_coverage": link_coverage,
                          "expected_link_ids": sorted(expected_link_ids),
                          "observed_link_ids": sorted(observables["link_flow_lps"])
                              if isinstance(observables.get("link_flow_lps"), dict) else [],
                          "has_link_data": observables.get("has_link_data")}
        if finite_out:
            for name in (*required_nodes, "link_flow_lps"):
                for series in observables[name].values():
                    try:
                        values = np.asarray(series, dtype=float)
                    except (TypeError, ValueError):
                        finite_out = False
                        break
                    if len(values) != step_count or not np.isfinite(values).all():
                        finite_out = False
                        break
        finite_coverage = bool(finite_labels and finite_out)
        if not node_coverage or not link_coverage:
            drainage_state = "unverified"
            drainage_manifest = {**drainage_manifest, "state": "unverified",
                                 "failed_checks": sorted(set(drainage_manifest.get("failed_checks", []))
                                                          | ({"incomplete_node_coverage"} if not node_coverage else set())
                                                          | ({"incomplete_link_coverage"} if not link_coverage else set()))}
    except Exception as exc:
        drainage_state = "unverified"
        drainage_manifest = {"policy_version": "tail_window_v1", "state": "unverified",
                             "failed_checks": [f"out_read_failed:{type(exc).__name__}"]}
        finite_coverage = False
        out_diagnostic = {"read_failed": type(exc).__name__}
        label_diagnostic = locals().get("label_diagnostic", {"unknown": True})

    source_initial = build_initial_condition_manifest(source_inp)
    effective_initial = build_initial_condition_manifest(effective_inp)
    if source_initial["state_sha256"] != effective_initial["state_sha256"]:
        raise ValueError("effective INP initial hydraulic state differs from the frozen source INP")
    return {
        "continuity_error_pct": continuity,
        "finite_coverage_ok": finite_coverage,
        "drainage_state": drainage_state,
        "drainage_manifest": drainage_manifest,
        "initial_condition": {
            "policy_id": source_initial["policy_id"], "state_sha256": source_initial["state_sha256"],
            "source_inp_sha256": source_initial["source_inp_sha256"],
            "effective_inp_sha256": effective_initial["source_inp_sha256"],
        },
        "flooding_report": report_manifest,
        "finite_coverage_diagnostics": {"rpt": label_diagnostic, "out": out_diagnostic},
    }


def stage_frozen_b_label_free(
    conn: sqlite3.Connection,
    experiment_root: str | Path,
    *,
    feature_contract_id: str,
    protocol_anchor_db_path: str | Path,
    manifest_name: str = "frozen_b_manifest.json",
    output_manifest_name: str = "experiment_source_manifest.json",
) -> dict:
    """Allocate fresh SQLite run IDs and return features without reading targets.

    `protocol.json` freezes network identity, scenario order, and scenario values
    before simulation. `frozen_b_manifest.json` binds to its byte hash and adds
    explicit artifact links and reconstructible hydraulic QA. The database must
    be empty. A hash link detects accidental changes; it is not a signature.
    """
    if conn.in_transaction:
        raise RuntimeError("staging owns its transaction; pass a connection with no active transaction")
    root = Path(experiment_root).resolve(strict=True)
    anchor_db_path = Path(protocol_anchor_db_path).resolve(strict=True)
    try:
        anchor_db_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("protocol anchor DB must remain outside experiment_root")
    manifest_path = root / manifest_name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read complete frozen B manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("frozen B manifest must be a JSON object")
    _reject_target_fields(manifest)
    if manifest.get("schema_version") != "frozen_b_transfer_manifest_v1" or manifest.get("status") != "complete":
        raise ValueError("unsupported or non-frozen B transfer manifest")
    if set(manifest) != {"schema_version", "status", "protocol_path", "protocol_sha256", "protocol_anchor", "runs"}:
        raise ValueError("frozen B manifest has missing or unsupported fields")
    protocol_relative = manifest["protocol_path"]
    if not isinstance(protocol_relative, str) or Path(protocol_relative).is_absolute():
        raise ValueError("protocol_path must be an explicit relative path")
    try:
        protocol_path = (root / protocol_relative).resolve(strict=True)
        protocol_path.relative_to(root)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read the frozen B protocol") from exc
    protocol_sha = _hash_bytes(protocol_path.read_bytes())
    if manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("frozen protocol SHA-256 does not match batch manifest")
    anchor_receipt = verify_b_transfer_protocol_anchor(
        protocol_anchor_db_path, protocol_path, manifest["protocol_anchor"]
    )
    if anchor_receipt["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("batch protocol hash differs from its pre-run external anchor")
    if not isinstance(protocol, dict) or protocol.get("schema_version") != "b_transfer_protocol_v1" or protocol.get("status") != "frozen_before_run":
        raise ValueError("unsupported or non-frozen B protocol")
    if set(protocol) != {"schema_version", "status", "network", "expected_scenario_keys", "scenarios",
                         "hydraulic_policy", "initial_condition"}:
        raise ValueError("frozen B protocol has missing or unsupported fields")
    if conn.execute("SELECT COUNT(*) FROM networks").fetchone()[0] or conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]:
        raise ValueError("B staging requires a fresh database with no networks or runs")
    if Path(output_manifest_name).name != output_manifest_name or (root / output_manifest_name).exists():
        raise FileExistsError("immutable B source manifest already exists or has an invalid name")
    if feature_contract_id not in {"tabular_v3_17", "tabular_v4_21"}:
        raise ValueError("feature_contract_id must be read from the selected candidate metadata")

    network_rec = _artifact(root, protocol["network"], "source network INP", allow_external=True)
    actual_source_initial = build_initial_condition_manifest(network_rec["path"])
    frozen_initial = protocol["initial_condition"]
    if (frozen_initial.get("source_inp_sha256") != network_rec["sha256"]
            or frozen_initial.get("state_sha256") != actual_source_initial["state_sha256"]):
        raise ValueError("source INP initial state differs from the pre-run protocol")
    policy = protocol["hydraulic_policy"]
    run_records = manifest["runs"]
    expected_keys = protocol["expected_scenario_keys"]
    if not isinstance(run_records, list) or not run_records or not isinstance(expected_keys, list):
        raise ValueError("frozen B manifest needs non-empty ordered scenarios")
    if any(not isinstance(key, str) or not key.strip() for key in expected_keys) or len(set(expected_keys)) != len(expected_keys):
        raise ValueError("expected_scenario_keys must be ordered, unique, non-empty strings")
    definitions = protocol["scenarios"]
    if not isinstance(definitions, list) or len(definitions) != len(expected_keys):
        raise ValueError("frozen protocol scenario definitions do not match its ordered keys")
    definitions_by_key = {}
    for definition in definitions:
        if not isinstance(definition, dict) or set(definition) != _PROTOCOL_SCENARIO_FIELDS:
            raise ValueError("each frozen scenario definition requires exact identity and timing fields")
        key = definition["scenario_key"]
        if not isinstance(key, str) or key in definitions_by_key:
            raise ValueError("frozen protocol scenario keys must be unique")
        definitions_by_key[key] = definition
    if list(definitions_by_key) != expected_keys:
        raise ValueError("frozen protocol scenario definitions are not in the declared stable order")
    by_key = {}
    for index, record in enumerate(run_records):
        if not isinstance(record, dict) or set(record) != _RUN_FIELDS:
            raise ValueError(f"run record {index} has unsupported or missing fields")
        key = record.get("scenario_key")
        if not isinstance(key, str) or key in by_key:
            raise ValueError("run scenario keys must be unique explicit strings")
        by_key[key] = record
    if set(by_key) != set(expected_keys) or len(run_records) != len(expected_keys):
        raise ValueError("run records do not match the frozen scenario-key list")

    prepared = []
    static_features = extract_static_features(network_rec["path"])
    if static_features.empty:
        raise ValueError("source network INP has no junctions")
    network_junction_ids = set(static_features["node_id"].astype(str))
    network_node_ids, network_link_ids = _inp_object_ids(network_rec["path"])
    if not network_junction_ids.issubset(network_node_ids):
        raise ValueError("source INP static junctions are inconsistent with hydraulic node inventory")
    for scenario_key in expected_keys:
        record = by_key[scenario_key]
        definition = definitions_by_key[scenario_key]
        shape_id = definition["shape_id"]
        if not isinstance(shape_id, str) or not shape_id.strip():
            raise ValueError(f"scenario {scenario_key} requires an exact shape_id")
        factor = _finite(definition["factor_mult"], f"{scenario_key}.factor_mult", minimum=0)
        duration = _finite(definition["duracion_horas"], f"{scenario_key}.duracion_horas", minimum=0)
        peak_time = _finite(definition["tiempo_al_pico_h"], f"{scenario_key}.tiempo_al_pico_h", minimum=0)
        if peak_time > duration:
            raise ValueError(f"scenario {scenario_key} peak time exceeds duration")
        declared_qa = record["hydraulic_quality"]
        if not isinstance(declared_qa, dict) or set(declared_qa) != _QA_FIELDS:
            raise ValueError(f"scenario {scenario_key} requires explicit QA declarations for comparison")
        artifacts = record["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != _ARTIFACT_ROLES:
            raise ValueError(f"scenario {scenario_key} must declare exactly five source artifact roles")
        checked = {role: _artifact(root, item, f"{scenario_key} {role}", allow_external=(role == "source_network_inp"))
                   for role, item in artifacts.items()}
        if (checked["source_network_inp"]["sha256"] != network_rec["sha256"]
                or checked["source_network_inp"]["path"] != network_rec["path"]):
            raise ValueError("source_network_inp role differs from frozen network INP")
        effective_static = extract_static_features(checked["effective_inp"]["path"])
        effective_node_ids, effective_link_ids = _inp_object_ids(checked["effective_inp"]["path"])
        if (effective_node_ids != network_node_ids or effective_link_ids != network_link_ids
                or set(effective_static["node_id"].astype(str)) != network_junction_ids):
            raise ValueError("effective INP junction coverage differs from the frozen source network")
        # Every QA evidence field is derived from the verified source artifacts;
        # caller-supplied values below are checked only as declarations.
        derived_qa = _derive_hydraulic_quality(
            source_inp=network_rec["path"], effective_inp=checked["effective_inp"]["path"],
            rpt_path=checked["swmm_rpt"]["path"], out_path=checked["swmm_out"]["path"],
            expected_node_ids=network_node_ids, expected_link_ids=network_link_ids, policy=policy,
        )
        if (declared_qa["max_continuity_error_pct"] != policy["max_continuity_error_pct"]
                or declared_qa["continuity_limit_rationale"] != policy["continuity_limit_rationale"]):
            raise ValueError(f"scenario {scenario_key} QA policy declaration differs from frozen protocol")
        if declared_qa["continuity_error_pct"] != derived_qa["continuity_error_pct"]:
            raise ValueError(f"scenario {scenario_key} continuity_error_pct declaration differs from RPT")
        for key in ("finite_coverage_ok", "drainage_state"):
            if declared_qa[key] != derived_qa[key]:
                raise ValueError(f"scenario {scenario_key} {key} declaration {declared_qa[key]!r} differs from derived {derived_qa[key]!r} using OUT/INP/RPT; details={derived_qa.get('finite_coverage_diagnostics')}")
        declared_initial = declared_qa["initial_condition"]
        if (not isinstance(declared_initial, dict) or set(declared_initial) != {"policy_id", "state_sha256"}
                or declared_initial.get("policy_id") != derived_qa["initial_condition"]["policy_id"]
                or declared_initial.get("state_sha256") != derived_qa["initial_condition"]["state_sha256"]):
            raise ValueError(f"scenario {scenario_key} initial_condition declaration differs from source/effective INP")
        if declared_qa["flooding_report"] != derived_qa["flooding_report"]:
            raise ValueError(f"scenario {scenario_key} flooding report declaration differs from parsed RPT")
        qa = {
            "continuity_error_pct": derived_qa["continuity_error_pct"],
            "max_continuity_error_pct": policy["max_continuity_error_pct"],
            "continuity_limit_rationale": policy["continuity_limit_rationale"],
            "finite_coverage_ok": derived_qa["finite_coverage_ok"],
            "drainage_state": derived_qa["drainage_state"],
            "initial_condition": derived_qa["initial_condition"],
            "flooding_report": derived_qa["flooding_report"],
        }
        # Build the stored eligibility from the derived values only.
        try:
            qa_decision = build_hydraulic_quality_decision(
                continuity_error_pct=qa["continuity_error_pct"], max_continuity_error_pct=policy["max_continuity_error_pct"],
                continuity_limit_rationale=policy["continuity_limit_rationale"],
                finite_coverage_ok=derived_qa["finite_coverage_ok"], drainage_state=derived_qa["drainage_state"],
                initial_condition=qa["initial_condition"], flooding_report_manifest=qa["flooding_report"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scenario {scenario_key} has invalid hydraulic QA") from exc
        features, node_ids, extraction = _make_features(network_rec["path"], checked["event_source"]["path"], feature_contract_id)
        if abs(extraction["duration_h"] - duration) > 1e-6 or abs(extraction["peak_time_h"] - peak_time) > 1e-6:
            raise ValueError(f"scenario {scenario_key} frozen duration/peak time disagrees with event series")
        if set(node_ids) != network_junction_ids:
            raise ValueError("feature node coverage does not match junction coverage in the frozen network")
        if not set(extraction["node_series"]).issubset(network_junction_ids):
            raise ValueError("event node coverage contains nodes outside the frozen network")
        prepared.append({
            "key": scenario_key, "record": record, "definition": definition, "artifacts": checked, "qa": qa,
            "qa_decision": qa_decision, "features": features, "node_ids": node_ids,
            "feature_data": extraction, "derived_qa": derived_qa,
        })

    source_manifest_path = root / output_manifest_name
    conn.execute("SAVEPOINT stage_frozen_b")
    wrote_manifest = False
    try:
        inp_bytes = network_rec["path"].read_bytes()
        inp = load_inp(network_rec["path"])
        flow_units = str(inp["OPTIONS"].get("FLOW_UNITS", "LPS") or "LPS")
        network_id = conn.execute(
            """INSERT INTO networks (network_sha256, name, source_filename, inp_bytes,
                       flow_units, created_at_utc) VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))""",
            (network_rec["sha256"], str(protocol["network"].get("name") or "B network"), network_rec["path"].name, inp_bytes, flow_units),
        ).lastrowid
        static_by_node = static_features.set_index(static_features["node_id"].astype(str))
        node_pk = {}
        for node_id in sorted(network_junction_ids):
            row = static_by_node.loc[node_id]
            node_pk[node_id] = conn.execute(
                """INSERT INTO nodes (network_id,node_id,node_type,coord_x,coord_y,
                           invert_elevation_m,max_depth_m,base_inflow_lps)
                   VALUES (?,?, 'junction',?,?,?,?,?)""",
                (network_id, node_id, _none(row.get("coord_x")), _none(row.get("coord_y")),
                 float(row["elev_fondo"]), float(row["prof_max"]), float(row["base_inflow_lps"])),
            ).lastrowid

        key_rows, feature_frames, run_ids, b_runs, planned_runs, out_runs = [], [], [], [], [], []
        for item in prepared:
            record = item["record"]
            definition = item["definition"]
            config = {
                "scenario_key": item["key"], "shape_id": definition["shape_id"],
                "factor_mult": float(definition["factor_mult"]), "duracion_horas": float(definition["duracion_horas"]),
                "tiempo_al_pico_h": float(definition["tiempo_al_pico_h"]),
                "event_series_sha256": item["artifacts"]["event_source"]["sha256"],
            }
            config_sha = _hash_json(config)
            scenario_id = conn.execute(
                """INSERT INTO scenarios (network_id,scenario_key,scenario_kind,factor_mult,
                           shape_id,duracion_horas,tiempo_al_pico_h,config_json,config_sha256)
                   VALUES (?,?, 'frozen_transfer',?,?,?,?,?,?)""",
                (network_id, item["key"], float(definition["factor_mult"]), definition["shape_id"],
                 float(definition["duracion_horas"]), float(definition["tiempo_al_pico_h"]), _canonical(config), config_sha),
            ).lastrowid
            run_id = conn.execute(
                """INSERT INTO runs (scenario_id,network_id,status,started_at_utc,completed_at_utc,
                           config_sha256,continuity_error_pct,node_count)
                   VALUES (?,?, 'COMPLETE',strftime('%Y-%m-%dT%H:%M:%SZ','now'),
                           strftime('%Y-%m-%dT%H:%M:%SZ','now'),?,?,?)""",
                (scenario_id, network_id, config_sha, item["qa_decision"]["continuity_error_pct"], len(item["node_ids"])),
            ).lastrowid
            qa = item["qa"]
            stored_quality = record_run_hydraulic_quality(
                conn, run_id=run_id, continuity_error_pct=item["qa_decision"]["continuity_error_pct"],
                max_continuity_error_pct=qa["max_continuity_error_pct"],
                continuity_limit_rationale=qa["continuity_limit_rationale"],
                finite_coverage_ok=qa["finite_coverage_ok"], drainage_state=qa["drainage_state"],
                initial_condition=qa["initial_condition"], flooding_report_manifest=qa["flooding_report"],
            )
            file_roles = {role: str(info["path"]) for role, info in item["artifacts"].items()}
            linked = register_run_source_artifacts(conn, run_id, file_roles)
            verify_run_source_artifacts(conn, run_id)
            if item["qa_decision"]["eligible"] and feature_contract_id == "tabular_v3_17":
                f = item["features"]
                values = []
                for row in item["feature_data"]["full"].itertuples(index=False):
                    feature = f.iloc[item["node_ids"].index(str(row.node_id))]
                    values.append((run_id, network_id, node_pk[str(row.node_id)],
                        feature["elev_fondo"], feature["prof_max"], int(feature["n_tuberias_in"]),
                        int(feature["n_tuberias_out"]), _none(feature["diam_max_in"]), _none(feature["diam_max_out"]),
                        _none(feature["pendiente_max_in"]), _none(feature["pendiente_out"]), feature["base_inflow_lps"],
                        _none(feature["dist_outfall_m"]), int(feature["n_nodos_aguas_arriba"]), feature["q_pico_acum_base"],
                        _none(feature["upstream_capacity_lps"]), feature["q_pico_nodo"], feature["q_pico_acum_escalado"],
                        feature["duracion_horas"], feature["tiempo_al_pico_h"], "tabular_v3_17"))
                conn.executemany("""INSERT INTO node_features (run_id,network_id,node_pk,elev_fondo,prof_max,
                    n_tuberias_in,n_tuberias_out,diam_max_in,diam_max_out,pendiente_max_in,pendiente_out,
                    base_inflow_lps,dist_outfall_m,n_nodos_aguas_arriba,q_pico_acum_base,upstream_capacity_lps,
                    q_pico_nodo,q_pico_acum_escalado,duracion_horas,tiempo_al_pico_h,feature_contract_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
            elif item["qa_decision"]["eligible"]:
                values = []
                for index, node_id in enumerate(item["node_ids"]):
                    feature = item["features"].iloc[index]
                    values.append((run_id, network_id, node_pk[node_id], feature["elev_fondo"], feature["prof_max"],
                        int(feature["n_tuberias_in"]), int(feature["n_tuberias_out"]), _none(feature["diam_max_in"]),
                        _none(feature["diam_max_out"]), _none(feature["pendiente_max_in"]), _none(feature["pendiente_out"]),
                        feature["base_inflow_lps"], _none(feature["dist_outfall_m"]), int(feature["n_nodos_aguas_arriba"]),
                        feature["q_pico_acum_base"], _none(feature["upstream_capacity_lps"]), feature["q_pico_nodo"],
                        feature["q_pico_acum_escalado"], feature["volumen_local_integrado_m3"], feature["volumen_tributario_integrado_m3"],
                        feature["duracion_efectiva_tributario_h"], feature["tiempo_primer_pico_tributario_h"],
                        _none(feature["tiempo_sobre_capacidad_h"]), _none(feature["integral_exceso_capacidad_m3"]), "tabular_v4_21"))
                conn.executemany("""INSERT INTO node_features_v4_21 (run_id,network_id,node_pk,elev_fondo,prof_max,
                    n_tuberias_in,n_tuberias_out,diam_max_in,diam_max_out,pendiente_max_in,pendiente_out,
                    base_inflow_lps,dist_outfall_m,n_nodos_aguas_arriba,q_pico_acum_base,upstream_capacity_lps,
                    q_pico_nodo,q_pico_acum_escalado,volumen_local_integrado_m3,volumen_tributario_integrado_m3,
                    duracion_efectiva_tributario_h,tiempo_primer_pico_tributario_h,tiempo_sobre_capacidad_h,
                    integral_exceso_capacidad_m3,feature_contract_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
            if item["qa_decision"]["eligible"]:
                keys = pd.DataFrame({
                    "run_id": run_id, "network_id": network_id, "scenario_id": scenario_id,
                    "scenario_key": item["key"], "scenario_kind": "frozen_transfer",
                    "node_id": item["node_ids"], "factor_mult": float(definition["factor_mult"]),
                    "shape_id": definition["shape_id"],
                }, columns=KEY_COLUMNS)
                key_rows.append(keys)
                feature_frames.append(item["features"].copy())
            run_ids.append(run_id)
            qa_row = conn.execute("SELECT manifest_json,policy_json,policy_sha256 FROM run_hydraulic_quality WHERE run_id=?", (run_id,)).fetchone()
            qa_manifest = json.loads(qa_row[0])
            qa_manifest["policy_sha256"] = qa_row[2]
            event_components = {"source_artifact": {
                "path": item["artifacts"]["event_source"]["path"].relative_to(root).as_posix(),
                "sha256": item["artifacts"]["event_source"]["sha256"],
                "byte_size": item["artifacts"]["event_source"]["byte_size"],
            }}
            event_series_sha256 = _hash_json(event_components)
            b_runs.append({"network_id": network_id, "run_id": run_id,
                "event_series_sha256": event_series_sha256, "hydraulic_quality": qa_manifest})
            planned_runs.append({"network_id": network_id, "run_id": run_id,
                "event_series_sha256": event_series_sha256, "event_series_components": event_components,
                "hydraulic_policy": json.loads(qa_row[1]), "hydraulic_policy_sha256": qa_row[2],
                "expected_node_ids": sorted(network_junction_ids)})
            out_runs.append({"run_id": run_id, "scenario_key": item["key"], "source_manifest_sha256": linked["source_manifest_sha256"],
                "scenario": {"scenario_kind": "frozen_transfer", "shape_id": definition["shape_id"],
                    "factor_mult": float(definition["factor_mult"]), "duracion_horas": float(definition["duracion_horas"]),
                    "tiempo_al_pico_h": float(definition["tiempo_al_pico_h"]), "config_sha256": config_sha},
                "eligible": item["qa_decision"]["eligible"], "reason_codes": item["qa_decision"]["reason_codes"],
                "hydraulic_quality": {key: value for key, value in stored_quality.items()
                                      if key not in {"initial_condition_json", "reason_codes_json", "manifest_json"}},
                "drainage_diagnostics": item["derived_qa"]["drainage_manifest"],
                "artifacts": {role: {"path": _manifest_path(info["path"], root), "sha256": info["sha256"], "byte_size": info["byte_size"]}
                              for role, info in item["artifacts"].items()}})
        if conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] != 0:
            raise ValueError("label-free staging must leave node_results empty")
        seal = {"schema_version": "label_free_b_source_manifest_v1",
            "input_manifest_path": manifest_path.name,
            "input_manifest_sha256": _hash_bytes(manifest_path.read_bytes()),
            "protocol_path": protocol_path.relative_to(root).as_posix(),
            "protocol_sha256": protocol_sha,
            "protocol_anchor": anchor_receipt,
            "protocol_anchor_db_path": str(anchor_db_path),
            "network_sha256": network_rec["sha256"], "feature_contract_id": feature_contract_id,
            "feature_contract_sha256": resolve_feature_contract(feature_contract_id).descriptor_sha256,
            "run_order": out_runs, "node_results_rows": 0}
        seal["manifest_sha256"] = _hash_json(seal)
        with source_manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            wrote_manifest = True
            json.dump(seal, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        conn.execute("RELEASE SAVEPOINT stage_frozen_b")
        empty_keys = pd.DataFrame(columns=KEY_COLUMNS)
        empty_features = pd.DataFrame(columns=resolve_feature_contract(feature_contract_id).feature_names)
        return {"network_id": network_id, "run_ids": run_ids,
            "keys": pd.concat(key_rows, ignore_index=True) if key_rows else empty_keys,
            "features": pd.concat(feature_frames, ignore_index=True) if feature_frames else empty_features,
            "feature_columns": list(resolve_feature_contract(feature_contract_id).feature_names),
            "feature_contract_id": feature_contract_id, "b_runs": b_runs,
            "planned_runs": planned_runs, "manifest": seal,
            "experiment_root": str(root), "manifest_path": str(source_manifest_path),
            "protocol_anchor_db_path": str(anchor_db_path)}
    except Exception:
        if wrote_manifest:
            source_manifest_path.unlink(missing_ok=True)
        conn.execute("ROLLBACK TO SAVEPOINT stage_frozen_b")
        conn.execute("RELEASE SAVEPOINT stage_frozen_b")
        raise


def _none(value):
    return None if pd.isna(value) else float(value)


def _manifest_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _verify_staged_state(conn: sqlite3.Connection, staged: dict) -> dict:
    manifest_path = Path(staged["manifest_path"]).resolve(strict=True)
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_digest = persisted_manifest.get("manifest_sha256")
    digest_payload = {key: value for key, value in persisted_manifest.items() if key != "manifest_sha256"}
    if manifest_digest != _hash_json(digest_payload) or persisted_manifest != staged.get("manifest"):
        raise ValueError("B source manifest seal mismatch")
    root = Path(staged["experiment_root"]).resolve(strict=True)
    if str(Path(staged["protocol_anchor_db_path"]).resolve(strict=True)) != persisted_manifest.get("protocol_anchor_db_path"):
        raise ValueError("protocol anchor database path differs from the sealed source manifest")
    input_manifest = (root / persisted_manifest["input_manifest_path"]).resolve(strict=True)
    input_manifest.relative_to(root)
    protocol_manifest = (root / persisted_manifest["protocol_path"]).resolve(strict=True)
    protocol_manifest.relative_to(root)
    verified_anchor = verify_b_transfer_protocol_anchor(
        staged["protocol_anchor_db_path"], protocol_manifest, persisted_manifest["protocol_anchor"]
    )
    if verified_anchor["protocol_sha256"] != persisted_manifest["protocol_sha256"]:
        raise ValueError("pre-run protocol anchor differs from the staged source manifest")
    if _hash_bytes(input_manifest.read_bytes()) != persisted_manifest["input_manifest_sha256"]:
        raise ValueError("frozen B manifest changed after staging")
    if _hash_bytes(protocol_manifest.read_bytes()) != persisted_manifest["protocol_sha256"]:
        raise ValueError("frozen B protocol changed after staging")
    expected_ids = [entry["run_id"] for entry in persisted_manifest["run_order"]]
    if staged.get("run_ids") != expected_ids:
        raise ValueError("staged run IDs differ from the sealed manifest order")
    network_rows = conn.execute(
        "SELECT network_id,network_sha256 FROM networks WHERE network_sha256=?",
        (persisted_manifest["network_sha256"],),
    ).fetchall()
    if len(network_rows) != 1:
        raise ValueError("staged B network identity is missing or ambiguous")
    actual_network_id, actual_network_hash = network_rows[0]
    if actual_network_hash != persisted_manifest["network_sha256"] or staged.get("network_id") != actual_network_id:
        raise ValueError("staged B network differs from the sealed source manifest")
    if staged.get("feature_contract_id") != persisted_manifest.get("feature_contract_id"):
        raise ValueError("staged feature contract differs from the sealed source manifest")
    root = Path(staged["experiment_root"]).resolve(strict=True)
    for entry in persisted_manifest["run_order"]:
        run_id = int(entry["run_id"])
        run = conn.execute(
            """SELECT r.network_id,s.scenario_key,r.status,r.config_sha256,s.scenario_kind,s.shape_id,
                      s.factor_mult,s.duracion_horas,s.tiempo_al_pico_h,s.config_json,s.config_sha256
               FROM runs r JOIN scenarios s ON s.scenario_id=r.scenario_id
               WHERE r.run_id=?""", (run_id,)
        ).fetchone()
        scenario = entry.get("scenario")
        expected_config = None
        if isinstance(scenario, dict):
            expected_config = {
                "scenario_key": entry["scenario_key"], "shape_id": scenario.get("shape_id"),
                "factor_mult": scenario.get("factor_mult"), "duracion_horas": scenario.get("duracion_horas"),
                "tiempo_al_pico_h": scenario.get("tiempo_al_pico_h"),
                "event_series_sha256": entry["artifacts"]["event_source"]["sha256"],
            }
        if (run is None or run[0] != actual_network_id or run[1] != entry["scenario_key"]
                or run[2] != "COMPLETE" or not isinstance(scenario, dict)
                or run[3] != scenario.get("config_sha256") or run[4] != scenario.get("scenario_kind")
                or run[5] != scenario.get("shape_id") or run[6] != scenario.get("factor_mult")
                or run[7] != scenario.get("duracion_horas") or run[8] != scenario.get("tiempo_al_pico_h")
                or run[10] != scenario.get("config_sha256") or run[10] != _hash_json(expected_config)
                or _canonical(json.loads(run[9])) != _canonical(expected_config)):
            raise ValueError(f"staged run identity or scenario metadata differs from the sealed manifest for run {run_id}")
        verified = verify_run_source_artifacts(conn, run_id)
        if verified["source_manifest_sha256"] != entry["source_manifest_sha256"]:
            raise ValueError(f"source manifest digest changed for B run {run_id}")
        for role, artifact in entry["artifacts"].items():
            registered = verified["artifacts"][role]
            recorded_path = _manifest_path(Path(registered["resolved_path"]), root)
            if (recorded_path != artifact["path"] or registered["sha256"] != artifact["sha256"]
                    or registered["byte_size"] != artifact["byte_size"]):
                raise ValueError(f"source artifact record changed for B run {run_id}, role {role}")
        quality = conn.execute(
            "SELECT manifest_json,policy_json,policy_sha256,created_at_utc FROM run_hydraulic_quality WHERE run_id=?",
            (run_id,)
        ).fetchone()
        quality_manifest = json.loads(quality[0])
        persisted_quality = {
            **quality_manifest, "policy": json.loads(quality[1]), "policy_sha256": quality[2],
            "created_at_utc": quality[3],
        }
        if (_canonical(persisted_quality) != _canonical(entry["hydraulic_quality"])
                or quality_manifest.get("eligible") is not entry["eligible"]
                or quality_manifest.get("reason_codes") != entry["reason_codes"]):
            raise ValueError(f"derived hydraulic QA differs from the sealed report for B run {run_id}")
    if conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] != 0:
        raise ValueError("B labels appeared before prediction; refusing label-free inference")
    return persisted_manifest


def register_and_predict_staged_transfer(
    candidate_dir: str | Path,
    domain_policy: dict,
    volume_bands_m3: list[float],
    output_root: str | Path,
    conn: sqlite3.Connection,
    staged: dict,
) -> Path:
    """Connect verified stage → preregistration → label-free prediction.

    Candidate metadata selects the feature contract. The preregistration plan is
    reconstructed from the sealed source manifest and SQLite policy/node rows;
    observed QA is still passed separately at prediction time.
    """
    from ..ml.bench import frozen_transfer
    from ..ml.bench.train import load_candidate_pair

    persisted = _verify_staged_state(conn, staged)
    _, _, candidate_meta = load_candidate_pair(candidate_dir, candidate_dir)
    if candidate_meta.get("feature_contract_id") != persisted["feature_contract_id"]:
        raise ValueError("staged feature contract differs from candidate metadata")
    planned_runs = []
    b_inp = None
    for entry in persisted["run_order"]:
        run_id = int(entry["run_id"])
        row = conn.execute(
            """SELECT r.network_id,q.policy_json,q.policy_sha256
               FROM runs r JOIN run_hydraulic_quality q ON q.run_id=r.run_id WHERE r.run_id=?""",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"staged run {run_id} has no persisted hydraulic policy")
        artifacts = entry["artifacts"]
        if b_inp is None:
            b_inp = str(Path(staged["experiment_root"]) / artifacts["source_network_inp"]["path"])
        elif Path(b_inp).resolve() != (Path(staged["experiment_root"]) / artifacts["source_network_inp"]["path"]).resolve():
            raise ValueError("staged runs do not share one frozen B INP")
        event_record = artifacts["event_source"]
        components = {"source_artifact": {
            "path": event_record["path"], "sha256": event_record["sha256"],
            "byte_size": event_record["byte_size"],
        }}
        nodes = [item[0] for item in conn.execute(
            "SELECT node_id FROM nodes WHERE network_id=? ORDER BY node_id", (row[0],)
        )]
        planned_runs.append({
            "network_id": int(row[0]), "run_id": run_id,
            "event_series_sha256": _hash_json(components), "event_series_components": components,
            "hydraulic_policy": json.loads(row[1]), "hydraulic_policy_sha256": row[2],
            "expected_node_ids": nodes,
        })
    if b_inp is None:
        raise ValueError("staged B has no run records")
    directory = frozen_transfer.register_transfer(
        candidate_dir, b_inp, domain_policy, planned_runs, volume_bands_m3, output_root
    )
    predict_staged_transfer(directory, conn, staged)
    return directory


def predict_staged_transfer(transfer_directory: str | Path, conn: sqlite3.Connection, staged: dict) -> None:
    """Re-verify all file hashes immediately before delegating to prediction."""
    from ..ml.bench.frozen_transfer import predict_transfer

    persisted_manifest = _verify_staged_state(conn, staged)
    # Rebuild from verified source paths so caller mutations to returned frames cannot
    # alter what gets predicted; then enforce full network junction coverage.
    keys, frames = [], []
    b_runs = []
    qa_by_run = {}
    for entry in persisted_manifest["run_order"]:
        run_id = int(entry["run_id"])
        qa_row = conn.execute(
            "SELECT manifest_json,policy_sha256 FROM run_hydraulic_quality WHERE run_id=?", (run_id,)
        ).fetchone()
        quality = json.loads(qa_row[0])
        quality["policy_sha256"] = qa_row[1]
        event = entry["artifacts"]["event_source"]
        components = {"source_artifact": {"path": event["path"], "sha256": event["sha256"], "byte_size": event["byte_size"]}}
        b_runs.append({"network_id": int(staged["network_id"]), "run_id": run_id,
                       "event_series_sha256": _hash_json(components), "hydraulic_quality": quality})
        qa_by_run[run_id] = quality
    for run_id in [int(entry["run_id"]) for entry in persisted_manifest["run_order"]]:
        if qa_by_run[run_id].get("eligible") is not True:
            continue
        scenario = conn.execute(
            """SELECT r.scenario_id,s.scenario_key,s.scenario_kind,s.shape_id,s.factor_mult
               FROM runs r JOIN scenarios s ON s.scenario_id=r.scenario_id
               WHERE r.run_id=?""", (run_id,)
        ).fetchone()
        links = verify_run_source_artifacts(conn, run_id)["artifacts"]
        f, node_ids, _ = _make_features(Path(links["source_network_inp"]["resolved_path"]),
                                        Path(links["event_source"]["resolved_path"]), staged["feature_contract_id"])
        feature_table = "node_features" if staged["feature_contract_id"] == "tabular_v3_17" else "node_features_v4_21"
        saved_ids = [row[0] for row in conn.execute(
            f"SELECT n.node_id FROM {feature_table} f JOIN nodes n ON n.node_pk=f.node_pk WHERE f.run_id=? ORDER BY n.node_id", (run_id,)
        )]
        if set(node_ids) != set(saved_ids) or len(node_ids) != len(saved_ids):
            raise ValueError(f"Incomplete node feature coverage for B run {run_id}")
        feature_columns = list(resolve_feature_contract(staged["feature_contract_id"]).feature_names)
        selected = ",".join(f"f.{column}" for column in feature_columns)
        persisted = conn.execute(
            f"SELECT n.node_id,{selected} FROM {feature_table} f JOIN nodes n ON n.node_pk=f.node_pk WHERE f.run_id=? ORDER BY n.node_id",
            (run_id,),
        ).fetchall()
        persisted_by_node = {str(row[0]): row[1:] for row in persisted}
        rebuilt_by_node = {node: f.iloc[index].to_numpy(dtype=float) for index, node in enumerate(node_ids)}
        for node_id in saved_ids:
            saved_values = np.asarray([np.nan if value is None else float(value) for value in persisted_by_node[node_id]])
            if not np.allclose(saved_values, rebuilt_by_node[node_id], rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError(f"Persisted B features differ from verified extractor output for run {run_id}, node {node_id}")
        by_node = {node: i for i, node in enumerate(node_ids)}
        ordered = [by_node[node] for node in saved_ids]
        f = f.iloc[ordered].reset_index(drop=True)
        keys.append(pd.DataFrame({
            "run_id": run_id, "network_id": int(staged["network_id"]),
            "scenario_id": int(scenario[0]), "scenario_key": scenario[1],
            "scenario_kind": scenario[2], "node_id": saved_ids,
            "factor_mult": float(scenario[4]), "shape_id": scenario[3],
        }, columns=KEY_COLUMNS))
        frames.append(f)
    if conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] != 0:
        raise ValueError("B labels appeared before prediction; refusing label-free inference")
    predict_transfer(transfer_directory,
                     pd.concat(keys, ignore_index=True) if keys else pd.DataFrame(columns=KEY_COLUMNS),
                     pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=staged["feature_columns"]),
                     b_runs)
