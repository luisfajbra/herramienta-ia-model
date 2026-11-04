"""Read-only checks for frozen sources before supervised multinet preparation.

This module does not build a dataset, extract labels, or certify that a source
is ready for training. Each verifier must describe its own coverage explicitly.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from .label_free_b_staging import _verify_staged_state
from .run_source_artifacts import verify_run_source_artifacts
from ..ml.contracts import resolve_feature_contract


def verify_label_free_source(database: str | Path, source_manifest: str | Path) -> dict:
    """Recheck frozen label-free source bindings and return per-run QA counts.

    This validates the existing source manifest, anchor, source artifacts,
    run identities and persisted QA with the production staging verifier. It
    does not re-extract labels or recompute features, and does not create a
    training dataset. A SHA-256 is an integrity check, not a signature.
    """
    db = Path(database).resolve(strict=True)
    manifest_path = Path(source_manifest).resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "label_free_b_source_manifest_v1":
        raise ValueError("expected a label-free Chico Norte source manifest")
    staged = {"manifest_path": str(manifest_path), "manifest": manifest,
              "experiment_root": str(manifest_path.parent),
              "protocol_anchor_db_path": manifest["protocol_anchor_db_path"],
              "run_ids": [item["run_id"] for item in manifest["run_order"]],
              "feature_contract_id": manifest["feature_contract_id"]}
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")  # Consistent read snapshot throughout verification.
        rows = conn.execute("SELECT network_id FROM networks WHERE network_sha256=?",
                            (manifest["network_sha256"],)).fetchall()
        if len(rows) != 1:
            raise ValueError("source network identity is missing or ambiguous")
        staged["network_id"] = rows[0][0]
        verified = _verify_staged_state(conn, staged)
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("source manifest changed during verification")
    runs = verified["run_order"]
    eligible = sum(item["eligible"] is True for item in runs)
    return {"database": str(db), "source_manifest": str(manifest_path),
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "network_sha256": verified["network_sha256"], "feature_contract_id": verified["feature_contract_id"],
            "feature_contract_sha256": verified["feature_contract_sha256"], "run_count": len(runs),
            "eligible_count": eligible, "excluded_count": len(runs) - eligible,
            "scope": "source bindings and QA; features and labels not revalidated"}


def verify_supervised_source(database: str | Path, supervised_manifest: str | Path) -> dict:
    """Verify a frozen supervised source and reproduce its stored RPT labels."""
    from ..extraction.labels import extract_labels

    db = Path(database).resolve(strict=True)
    path = Path(supervised_manifest).resolve(strict=True)
    payload = path.read_bytes()
    final = json.loads(payload)
    canonical = json.dumps({k: v for k, v in final.items() if k != "manifest_sha256"},
                           sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if (final.get("schema_version") != "a_supervised_source_manifest_v1"
            or final.get("status") != "complete"
            or hashlib.sha256(canonical).hexdigest() != final.get("manifest_sha256")):
        raise ValueError("supervised source manifest seal mismatch")
    source_path = (path.parent / final["source_manifest_path"]).resolve(strict=True)
    source_path.relative_to(path.parent)
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != final["source_manifest_sha256"]:
        raise ValueError("supervised staging source manifest hash mismatch")
    source = json.loads(source_bytes)
    for field in ("network_sha256", "protocol_sha256", "feature_contract_id", "feature_contract_sha256", "protocol_anchor_db_path"):
        if final[field] != source[field]:
            raise ValueError(f"supervised source differs from staging: {field}")
    entries = source["run_order"]
    staged = {"manifest_path": str(source_path), "manifest": source,
              "experiment_root": str(path.parent), "protocol_anchor_db_path": source["protocol_anchor_db_path"],
              "run_ids": [entry["run_id"] for entry in entries], "feature_contract_id": source["feature_contract_id"]}
    if final["run_ids"] != staged["run_ids"]:
        raise ValueError("supervised run order differs from staging")
    eligible = [e for e in entries if e["eligible"] is True]
    if final["eligible_run_ids"] != [e["run_id"] for e in eligible]:
        raise ValueError("supervised eligible runs differ from staging")
    threshold = float(final["flood_threshold_m3"])
    count = 0
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")  # Consistent read snapshot throughout verification.
        networks = conn.execute("SELECT network_id FROM networks WHERE network_sha256=?", (source["network_sha256"],)).fetchall()
        if len(networks) != 1:
            raise ValueError("source network identity is missing or ambiguous")
        staged["network_id"] = networks[0][0]
        _verify_staged_state(conn, staged, allow_supervised=True)
        nodes = [r[0] for r in conn.execute("SELECT node_id FROM nodes WHERE network_id=? ORDER BY node_id", (staged["network_id"],))]
        for entry in eligible:
            run_id = entry["run_id"]
            artifacts = verify_run_source_artifacts(conn, run_id)["artifacts"]
            labels = extract_labels(Path(artifacts["swmm_rpt"]["resolved_path"]), nodes, threshold_m3=threshold)
            if labels.attrs.get("flooding_report_manifest", {}).get("report_sha256") != artifacts["swmm_rpt"]["sha256"]:
                raise ValueError(f"label report hash mismatch for run {run_id}")
            expected = {str(row.node_id): (int(row.inunda), float(row.vol_inundacion_m3)) for row in labels.itertuples()}
            stored = {str(n): (int(c), float(v)) for n, c, v in conn.execute(
                "SELECT n.node_id,r.inunda,r.vol_inundacion_m3 FROM node_results r JOIN nodes n ON n.node_pk=r.node_pk AND n.network_id=r.network_id WHERE r.run_id=? AND r.network_id=?",
                (run_id, staged["network_id"]))}
            if len(expected) != len(nodes) or set(expected) != set(nodes) or stored != expected:
                raise ValueError(f"stored target labels differ from report for run {run_id}")
            count += len(stored)
        if count != final["node_results_rows"] or conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] != count:
            raise ValueError("supervised target label count differs from manifest")
    if source_path.read_bytes() != source_bytes or path.read_bytes() != payload:
        raise ValueError("supervised source manifest changed during verification")
    return {"database": str(db), "source_manifest": str(path),
            "source_manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "network_sha256": source["network_sha256"], "feature_contract_id": source["feature_contract_id"],
            "feature_contract_sha256": source["feature_contract_sha256"], "run_count": len(entries),
            "eligible_count": len(eligible), "excluded_count": len(entries) - len(eligible),
            "target_rows_verified": count, "scope": "source bindings, QA, and stored targets; features not recomputed"}


def verify_source_pair(a_database: str | Path, a_manifest: str | Path,
                       b_database: str | Path, b_manifest: str | Path) -> dict:
    """Verify both sources and pair compatibility without materializing rows."""
    a = verify_supervised_source(a_database, a_manifest)
    b = verify_label_free_source(b_database, b_manifest)
    if a["network_sha256"] == b["network_sha256"]:
        raise ValueError("source networks have the same INP hash")
    if a["feature_contract_id"] != "tabular_v4_21" or b["feature_contract_id"] != "tabular_v4_21":
        raise ValueError("sources do not share the approved feature contract")
    descriptor = resolve_feature_contract("tabular_v4_21").descriptor_sha256
    if a["feature_contract_sha256"] != descriptor or b["feature_contract_sha256"] != descriptor:
        raise ValueError("source feature contract descriptors differ from the approved contract")
    if a["eligible_count"] < 1 or b["eligible_count"] < 1:
        raise ValueError("both source networks require at least one QA-eligible run")
    return {"chico_sur": a, "chico_norte": b,
            "scope": "pair identity and individual source verification; features not recomputed; no joint dataset"}


def _verify_staged_features(database: str | Path, source_manifest: str | Path, *, supervised: bool) -> dict:
    import numpy as np
    from .label_free_b_staging import _make_features

    db = Path(database).resolve(strict=True)
    manifest_path = Path(source_manifest).resolve(strict=True)
    payload = manifest_path.read_bytes()
    source = json.loads(payload)
    schema = "a_verified_source_manifest_v1" if supervised else "label_free_b_source_manifest_v1"
    if source.get("schema_version") != schema:
        raise ValueError("unexpected staged source manifest schema")
    contract = resolve_feature_contract("tabular_v4_21")
    if (source["feature_contract_id"] != contract.contract_id
            or source["feature_contract_sha256"] != contract.descriptor_sha256):
        raise ValueError("feature contract does not match approved descriptor")
    staged = {"manifest_path": str(manifest_path), "manifest": source,
              "experiment_root": str(manifest_path.parent), "protocol_anchor_db_path": source["protocol_anchor_db_path"],
              "run_ids": [entry["run_id"] for entry in source["run_order"]], "feature_contract_id": contract.contract_id}
    checked = 0
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")
        networks = conn.execute("SELECT network_id FROM networks WHERE network_sha256=?", (source["network_sha256"],)).fetchall()
        if len(networks) != 1:
            raise ValueError("source network identity is missing or ambiguous")
        staged["network_id"] = networks[0][0]
        _verify_staged_state(conn, staged, allow_supervised=supervised)
        if conn.execute("SELECT COUNT(*) FROM node_features_v4_21 WHERE feature_contract_id IS NULL OR feature_contract_id != ?", (contract.contract_id,)).fetchone()[0]:
            raise ValueError("feature row contract differs from the approved contract")
        inventory = [str(row[0]) for row in conn.execute("SELECT node_id FROM nodes WHERE network_id=? ORDER BY node_id", (staged["network_id"],))]
        columns = list(contract.feature_names)
        selected = ",".join(f"f.{column}" for column in columns)
        for entry in source["run_order"]:
            run_id = entry["run_id"]
            rows = conn.execute(
                f"SELECT n.node_id,{selected} FROM node_features_v4_21 f JOIN nodes n ON n.node_pk=f.node_pk AND n.network_id=f.network_id WHERE f.run_id=? ORDER BY n.node_id", (run_id,)).fetchall()
            if entry["eligible"] is not True:
                if rows:
                    raise ValueError(f"ineligible run {run_id} unexpectedly contains features")
                continue
            artifacts = verify_run_source_artifacts(conn, run_id)["artifacts"]
            rebuilt, ids, _ = _make_features(Path(artifacts["source_network_inp"]["resolved_path"]),
                                              Path(artifacts["event_source"]["resolved_path"]), contract.contract_id)
            saved_ids = [str(row[0]) for row in rows]
            if saved_ids != inventory or len(ids) != len(set(ids)) or set(ids) != set(inventory):
                raise ValueError(f"feature node coverage mismatch for run {run_id}")
            calculated = {str(node): rebuilt.iloc[index].to_numpy(dtype=float) for index, node in enumerate(ids)}
            for row in rows:
                values = np.asarray([np.nan if value is None else float(value) for value in row[1:]])
                if not np.allclose(values, calculated[str(row[0])], rtol=1e-12, atol=1e-12, equal_nan=True):
                    raise ValueError(f"stored feature differs from verified source for run {run_id}, node {row[0]}")
            checked += len(rows)
        if conn.execute("SELECT COUNT(*) FROM node_features_v4_21").fetchone()[0] != checked:
            raise ValueError("feature table contains rows outside the verified eligible runs")
    if manifest_path.read_bytes() != payload:
        raise ValueError("source manifest changed during feature verification")
    return {"network_sha256": source["network_sha256"], "feature_rows_verified": checked,
            "scope": "eligible features reconstructed from source; no training"}


def verify_label_free_features(database: str | Path, source_manifest: str | Path) -> dict:
    """Rebuild and compare eligible v4 features from linked network/event files."""
    return _verify_staged_features(database, source_manifest, supervised=False)


def verify_supervised_features(database: str | Path, supervised_manifest: str | Path) -> dict:
    """Verify supervised labels and rebuild features from sealed source files."""
    final_path = Path(supervised_manifest).resolve(strict=True)
    payload = final_path.read_bytes()
    verify_supervised_source(database, final_path)
    final = json.loads(payload)
    staged_path = (final_path.parent / final["source_manifest_path"]).resolve(strict=True)
    staged_path.relative_to(final_path.parent)
    result = _verify_staged_features(database, staged_path, supervised=True)
    if final_path.read_bytes() != payload:
        raise ValueError("supervised manifest changed during feature verification")
    return result


def extract_label_free_targets(database: str | Path, source_manifest: str | Path, *, flood_threshold_m3: float) -> dict:
    """Extract eligible labels into memory from sealed reports; never persist them."""
    import pandas as pd
    from ..extraction.labels import extract_labels
    from ..ml.bench.frozen_transfer import _hash

    if float(flood_threshold_m3) != 1.0:
        raise ValueError("joint development currently requires the approved 1 m3 flood threshold")
    db, manifest_path = Path(database).resolve(strict=True), Path(source_manifest).resolve(strict=True)
    payload = manifest_path.read_bytes()
    source = json.loads(payload)
    if source.get("schema_version") != "label_free_b_source_manifest_v1":
        raise ValueError("expected a label-free source manifest")
    staged = {"manifest_path": str(manifest_path), "manifest": source,
              "experiment_root": str(manifest_path.parent), "protocol_anchor_db_path": source["protocol_anchor_db_path"],
              "run_ids": [entry["run_id"] for entry in source["run_order"]], "feature_contract_id": source["feature_contract_id"]}
    labels, exclusions = [], []
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")
        networks = conn.execute("SELECT network_id FROM networks WHERE network_sha256=?", (source["network_sha256"],)).fetchall()
        if len(networks) != 1:
            raise ValueError("source network identity is missing or ambiguous")
        staged["network_id"] = networks[0][0]
        _verify_staged_state(conn, staged)
        nodes = [str(row[0]) for row in conn.execute("SELECT node_id FROM nodes WHERE network_id=? ORDER BY node_id", (staged["network_id"],))]
        for entry in source["run_order"]:
            run_id = int(entry["run_id"])
            if entry["eligible"] is not True:
                exclusions.append({"run_id": run_id, "scenario_key": entry["scenario_key"], "reason_codes": entry["reason_codes"]})
                continue
            rpt = verify_run_source_artifacts(conn, run_id)["artifacts"]["swmm_rpt"]
            if entry["hydraulic_quality"]["flooding_report"]["report_sha256"] != rpt["sha256"]:
                raise ValueError(f"QA report hash mismatch for run {run_id}")
            extracted = extract_labels(Path(rpt["resolved_path"]), nodes, threshold_m3=1.0)
            if extracted.attrs.get("flooding_report_manifest", {}).get("report_sha256") != rpt["sha256"]:
                raise ValueError(f"target report hash mismatch for run {run_id}")
            if extracted["node_id"].astype(str).duplicated().any() or set(extracted["node_id"].astype(str)) != set(nodes):
                raise ValueError(f"target node coverage mismatch for run {run_id}")
            frame = extracted.copy()
            frame["node_id"] = frame["node_id"].astype(str)
            frame["network_id"], frame["run_id"] = int(staged["network_id"]), run_id
            labels.append(frame[["network_id", "run_id", "node_id", "inunda", "vol_inundacion_m3"]])
    if not labels:
        raise ValueError("no eligible runs remain for supervised targets")
    if manifest_path.read_bytes() != payload:
        raise ValueError("source manifest changed during target extraction")
    frame = pd.concat(labels, ignore_index=True)
    keys = ["network_id", "run_id", "node_id"]
    if frame.duplicated(keys).any():
        raise ValueError("target keys are duplicated")
    return {"labels": frame, "labels_sha256": _hash(frame.sort_values(keys).to_dict(orient="records")),
            "excluded_runs": exclusions, "source_manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "scope": "in-memory target extraction; no model training or database writes"}


def plan_joint_identities(a_conn: sqlite3.Connection, b_conn: sqlite3.Connection) -> dict:
    """Plan distinct destination keys; callers must independently verify sources.

    Source IDs are local to each connection. No database is changed and this
    plan alone does not validate file provenance, features, QA, or labels.
    """
    result = {}
    counters = {"nodes": 0, "scenarios": 0, "runs": 0}
    seen_hashes = set()
    for index, (name, conn) in enumerate((("chico_sur", a_conn), ("chico_norte", b_conn)), start=1):
        networks = conn.execute("SELECT network_id, network_sha256 FROM networks").fetchall()
        if len(networks) != 1 or networks[0][1] in seen_hashes:
            raise ValueError("each source must contain one distinct network")
        source_network, digest = networks[0]
        seen_hashes.add(digest)
        entry = {"source_network_id": source_network, "network_id": index, "network_sha256": digest}
        source_ids = {}
        for table, pk in (("nodes", "node_pk"), ("scenarios", "scenario_id"), ("runs", "run_id")):
            rows = conn.execute(f"SELECT {pk}, network_id FROM {table} ORDER BY {pk}").fetchall()
            ids = [row[0] for row in rows]
            if len(ids) != len(set(ids)) or any(row[1] != source_network for row in rows):
                raise ValueError(f"{name} {table} keys are duplicate or belong to another network")
            source_ids[table] = set(ids)
            mapping = {}
            for old in ids:
                counters[table] += 1
                mapping[old] = counters[table]
            entry[table] = mapping
        if any(row[0] not in source_ids["scenarios"] for row in conn.execute("SELECT scenario_id FROM runs")):
            raise ValueError(f"{name} run points to an unknown scenario")
        result[name] = entry
    return result
