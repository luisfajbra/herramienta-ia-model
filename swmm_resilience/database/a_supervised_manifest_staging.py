"""Build supervised A rows from a frozen, fully linked SWMM run manifest.

Runs, independently recalculated hydraulic quality, and source links are first
staged through the v2 manifest verifier. Eligible targets are then extracted
from each exact run's verified RPT in a second transaction. A crash between
those phases leaves a source-only database without the final A manifest; such
a database is intentionally not a training input and must be rebuilt fresh.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3

import yaml

from ..config import load_config
from ..extraction.labels import extract_labels
from ..ml.contracts import resolve_feature_contract
from .b_transfer_protocol import verify_b_transfer_protocol_anchor
from .label_free_b_staging import stage_frozen_b_label_free
from .run_source_artifacts import verify_run_source_artifacts


_ARTIFACT_ROLES = frozenset({
    "source_network_inp", "effective_inp", "event_source", "swmm_rpt", "swmm_out",
})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_config(config_path: Path):
    try:
        payload = config_path.read_bytes()
        config = load_config(str(config_path))
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, KeyError, ValueError) as exc:
        raise ValueError("config must load through the official config loader") from exc
    threshold = float(config.dataset.flood_threshold_m3)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("dataset.flood_threshold_m3 must be finite and non-negative")
    return config, threshold, _sha256(payload)


def _approved_a_scenario_keys(config) -> list[str]:
    approved_factors = tuple(round(0.2 * index, 1) for index in range(1, 26))
    factors = tuple(float(value) for value in config.factors())
    if factors != approved_factors:
        raise ValueError("A training requires the approved 25-factor grid from 0.2 through 5.0 by 0.2")
    shapes_dir = config.simulation.hydrograph_shapes_dir
    shape_ids = sorted(path.stem for path in shapes_dir.glob("*.csv"))
    if len(shape_ids) != 39 or len(set(shape_ids)) != 39:
        raise ValueError("A training requires exactly 39 uniquely named hydrograph shape CSVs")
    keys = [f"A:base:f{factor:.3f}" for factor in approved_factors]
    keys.extend(f"A:{shape_id}:f{factor:.3f}" for shape_id in shape_ids for factor in approved_factors)
    return keys


def _write_manifest_exclusive(path: Path, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)


def _require_fresh_database(conn: sqlite3.Connection) -> None:
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ) if row[0] not in {"schema_migrations", "schema_migration_validators"}]
    for table in tables:
        quoted = '"' + table.replace('"', '""') + '"'
        if conn.execute(f"SELECT 1 FROM {quoted} LIMIT 1").fetchone() is not None:
            raise ValueError(f"A staging requires a fresh database; table {table} is not empty")


def stage_frozen_a_supervised(
    conn: sqlite3.Connection,
    experiment_root: str | Path,
    *,
    protocol_anchor_db_path: str | Path,
    config_path: str | Path,
    manifest_name: str = "frozen_a_training_manifest.json",
    staging_manifest_name: str = "a_verified_source_manifest.json",
    output_manifest_name: str = "a_training_source_manifest.json",
) -> dict:
    """Stage all A run provenance and persist labels only for QA-eligible runs.

    The caller supplies a fresh, migrated database. The protocol v2 anchor,
    expected scenario order, five role links, file hashes, effective-INP/event
    equivalence, and hydraulic QA are verified by the shared frozen-manifest
    stager. The A manifest seals the final eligible/excluded run IDs and label
    counts only after target insertion commits.
    """
    if conn.in_transaction:
        raise RuntimeError("A staging owns its transactions; pass a connection with no active transaction")
    root = Path(experiment_root).resolve(strict=True)
    config = Path(config_path).resolve(strict=True)
    output_path = root / output_manifest_name
    if (Path(manifest_name).name != manifest_name
            or Path(staging_manifest_name).name != staging_manifest_name
            or Path(output_manifest_name).name != output_manifest_name):
        raise ValueError("manifest names must be simple filenames inside experiment_root")
    if len({manifest_name, staging_manifest_name, output_manifest_name}) != 3:
        raise ValueError("input, staging, and final A manifest names must be distinct")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("final A training manifest already exists")
    _require_fresh_database(conn)
    config_object, threshold, config_sha256 = _read_config(config)
    approved_keys = _approved_a_scenario_keys(config_object)
    contract = resolve_feature_contract("tabular_v4_21")

    staged = stage_frozen_b_label_free(
        conn, root, feature_contract_id="tabular_v4_21",
        protocol_anchor_db_path=protocol_anchor_db_path,
        manifest_name=manifest_name,
        output_manifest_name=staging_manifest_name,
        expected_network_role="A",
        expected_manifest_schema="frozen_a_training_manifest_v1",
        expected_network_path=config_object.network.inp_path,
        expected_scenario_keys=approved_keys,
    )
    stage_manifest_path = Path(staged["manifest_path"]).resolve(strict=True)
    staged_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    if staged_manifest.get("manifest_sha256") != hashlib.sha256(
        _canonical({key: value for key, value in staged_manifest.items() if key != "manifest_sha256"}).encode("utf-8")
    ).hexdigest():
        raise ValueError("verified A source manifest seal is invalid")
    protocol_path = (root / staged_manifest["protocol_path"]).resolve(strict=True)
    protocol_path.relative_to(root)
    protocol_sha256 = _sha256(protocol_path.read_bytes())
    if protocol_sha256 != staged_manifest.get("protocol_sha256"):
        raise ValueError("v2 A protocol changed after source staging")
    anchor = verify_b_transfer_protocol_anchor(
        protocol_anchor_db_path, protocol_path, staged_manifest["protocol_anchor"]
    )
    if anchor["protocol_sha256"] != protocol_sha256:
        raise ValueError("v2 A protocol anchor does not match the frozen protocol")
    input_manifest_path = (root / staged_manifest["input_manifest_path"]).resolve(strict=True)
    input_manifest_path.relative_to(root)
    if _sha256(input_manifest_path.read_bytes()) != staged_manifest["input_manifest_sha256"]:
        raise ValueError("complete A run manifest changed after source staging")
    entries = staged_manifest.get("run_order")
    if not isinstance(entries, list) or [entry.get("run_id") for entry in entries] != staged["run_ids"]:
        raise ValueError("verified A source manifest run order differs from allocated run IDs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if [entry.get("scenario_key") for entry in entries] != protocol.get("expected_scenario_keys"):
        raise ValueError("verified A run records differ from frozen expected scenario-key order")
    if conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] != 0:
        raise ValueError("fresh A staging database unexpectedly contains node targets")

    target_rows = []
    excluded = []
    eligible_ids = []
    report_records = []
    for entry in entries:
        run_id = int(entry["run_id"])
        scenario_key = entry["scenario_key"]
        quality = conn.execute(
            "SELECT manifest_json FROM run_hydraulic_quality WHERE run_id=?", (run_id,)
        ).fetchone()
        if quality is None:
            raise ValueError(f"A run {run_id} has no persisted hydraulic QA")
        quality_manifest = json.loads(quality[0])
        eligible = quality_manifest.get("eligible")
        reason_codes = quality_manifest.get("reason_codes")
        if type(eligible) is not bool or not isinstance(reason_codes, list):
            raise ValueError(f"A run {run_id} has malformed persisted QA")

        source = verify_run_source_artifacts(conn, run_id)
        artifacts = source.get("artifacts", {})
        if set(artifacts) != _ARTIFACT_ROLES:
            raise ValueError(f"A run {run_id} does not have exactly five linked source artifacts")
        sealed_artifacts = entry.get("artifacts", {})
        if set(sealed_artifacts) != _ARTIFACT_ROLES:
            raise ValueError(f"A run {run_id} source manifest does not contain five artifact roles")
        for role, item in artifacts.items():
            path = Path(item["resolved_path"]).resolve(strict=True)
            if not path.is_file() or _sha256(path.read_bytes()) != item["sha256"]:
                raise ValueError(f"A run {run_id} artifact changed after source staging: {role}")
            sealed = sealed_artifacts[role]
            if (sealed.get("path") != _manifest_path(path, root)
                    or sealed.get("sha256") != item["sha256"]
                    or sealed.get("byte_size") != item["byte_size"]):
                raise ValueError(f"A run {run_id} artifact differs from its sealed source record: {role}")

        if not eligible:
            excluded.append({"run_id": run_id, "scenario_key": scenario_key,
                             "reason_codes": sorted(set(reason_codes))})
            report_records.append({"run_id": run_id, "scenario_key": scenario_key,
                                   "eligible": False, "reason_codes": sorted(set(reason_codes)),
                                   "node_results_rows": 0})
            continue
        if reason_codes:
            raise ValueError(f"A run {run_id} is eligible but contains QA exclusion reasons")

        rpt = Path(artifacts["swmm_rpt"]["resolved_path"])
        nodes = [row[0] for row in conn.execute(
            "SELECT node_id FROM nodes WHERE network_id=? ORDER BY node_id", (staged["network_id"],)
        )]
        labels = extract_labels(rpt, nodes, threshold_m3=threshold)
        label_manifest = labels.attrs.get("flooding_report_manifest", {})
        qa_report = quality_manifest.get("flooding_report", {})
        if (not isinstance(label_manifest, dict)
                or label_manifest.get("report_sha256") != artifacts["swmm_rpt"]["sha256"]
                or label_manifest.get("report_sha256") != qa_report.get("report_sha256")):
            raise ValueError(f"A run {run_id} labels do not come from the QA-validated RPT bytes")
        rows = conn.execute(
            "SELECT node_pk,node_id FROM nodes WHERE network_id=? ORDER BY node_id", (staged["network_id"],)
        ).fetchall()
        label_by_node = labels.set_index("node_id")
        if set(label_by_node.index.astype(str)) != {str(row[1]) for row in rows}:
            raise ValueError(f"A run {run_id} report labels do not cover the full network node inventory")
        for node_pk, node_id in rows:
            label = label_by_node.loc[str(node_id)]
            target_rows.append((run_id, staged["network_id"], node_pk, int(label["inunda"]),
                                float(label["vol_inundacion_m3"])))
        eligible_ids.append(run_id)
        report_records.append({"run_id": run_id, "scenario_key": scenario_key, "eligible": True,
                               "reason_codes": [], "node_results_rows": len(rows),
                               "rpt_sha256": artifacts["swmm_rpt"]["sha256"]})

    if not eligible_ids:
        raise ValueError("A staging produced no QA-eligible runs; no training manifest will be written")
    if conn.in_transaction:
        raise RuntimeError("unexpected open transaction after verified source staging")
    final = {
        "schema_version": "a_supervised_source_manifest_v1",
        "status": "complete",
        "source_manifest_path": stage_manifest_path.name,
        "source_manifest_sha256": _sha256(stage_manifest_path.read_bytes()),
        "protocol_path": staged_manifest["protocol_path"],
        "protocol_sha256": staged_manifest["protocol_sha256"],
        "protocol_anchor": staged_manifest["protocol_anchor"],
        "protocol_anchor_db_path": str(Path(protocol_anchor_db_path).resolve(strict=True)),
        "network_sha256": staged_manifest["network_sha256"],
        "feature_contract_id": contract.contract_id,
        "feature_contract_sha256": contract.descriptor_sha256,
        "config_sha256": config_sha256,
        "flood_threshold_m3": threshold,
        "run_ids": staged["run_ids"],
        "eligible_run_ids": eligible_ids,
        "excluded_runs": excluded,
        "node_results_rows": len(target_rows),
        "runs": report_records,
    }
    final["manifest_sha256"] = _sha256(_canonical(final).encode("utf-8"))
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] != 0:
            raise ValueError("A target table changed before the label transaction")
        conn.executemany(
            """INSERT INTO node_results (run_id,network_id,node_pk,inunda,vol_inundacion_m3,
                       peak_flooding_lps,flooding_duration_min,max_depth_m,max_depth_ratio)
               VALUES (?,?,?,?,?,NULL,NULL,NULL,NULL)""",
            target_rows,
        )
        # Publish the seal before committing targets. If publication fails,
        # the transaction rolls back and no final manifest exists. If the
        # process dies after publication but before commit, loader cardinality
        # validation rejects the source-only database.
        _write_manifest_exclusive(output_path, final)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"run_ids": staged["run_ids"], "eligible_run_ids": eligible_ids,
            "excluded_runs": excluded, "node_results_rows": len(target_rows),
            "manifest": final, "manifest_path": str(output_path),
            "staging_manifest_path": str(stage_manifest_path), "threshold_m3": threshold}
