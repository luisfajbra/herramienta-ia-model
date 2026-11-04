"""Building blocks for a supervised multinet dataset; no public builder yet.

Inputs to the private copy helper must first be independently verified against
their anchored manifests. This module alone does not admit a dataset for ML.
"""
from __future__ import annotations

import math
import hashlib
import json
import os
import platform
from pathlib import Path
import sqlite3
from importlib.metadata import PackageNotFoundError, version

from .run_source_artifacts import register_run_source_artifacts, verify_run_source_artifacts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_no_wal(path: Path, label: str) -> None:
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"{label} has a nonempty WAL and cannot be verified by file hash")


def _implementation_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    library = root / "swmm_resilience"
    files = sorted(p for p in library.rglob("*") if p.is_file() and p.suffix in {".py", ".sql"})
    for name in ("config.yaml", "pyproject.toml", "requirements.txt"):
        path = root / name
        if path.is_file():
            files.append(path)
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "swmm-api", "pyswmm", "swmm-toolkit"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"python_version": platform.python_version(), "packages": versions,
            "files": {path.relative_to(root).as_posix(): _file_sha256(path) for path in files}}


def _snapshot_database(source: Path, destination: Path) -> None:
    """Create a consistent SQLite snapshot without modifying the source."""
    from contextlib import closing
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with closing(sqlite3.connect(source.resolve(strict=True).as_uri() + "?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(destination)) as snapshot:
            original.backup(snapshot)
            if snapshot.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError(f"source snapshot integrity check failed: {source}")


def _write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def materialize_joint_dataset(a_database, a_manifest, b_database, b_manifest, output_directory,
                              *, expected_b_labels_sha256: str, expected_network_sha256s: tuple[str, str]) -> dict:
    """Materialize a new, verified source package; it is not yet training-ready.

    The caller explicitly supplies expected network identities and the previously
    frozen Chico Norte label digest. Originals are opened read-only. Existing
    outputs are never reused; failures leave a failed status, not a complete seal.
    """
    from contextlib import closing
    import pandas as pd
    from .connection import connect_managed_database
    from .migrations import apply_migrations
    from .joint_preflight import (verify_source_pair, verify_supervised_features,
                                  verify_label_free_features, extract_label_free_targets,
                                  plan_joint_identities)

    def digest_ok(value):
        return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)
    if (not digest_ok(expected_b_labels_sha256) or not isinstance(expected_network_sha256s, tuple)
            or len(expected_network_sha256s) != 2 or not all(digest_ok(x) for x in expected_network_sha256s)
            or expected_network_sha256s[0] == expected_network_sha256s[1]):
        raise ValueError("explicit distinct network and label SHA-256 values are required")
    source_a, manifest_a = Path(a_database).resolve(strict=True), Path(a_manifest).resolve(strict=True)
    source_b, manifest_b = Path(b_database).resolve(strict=True), Path(b_manifest).resolve(strict=True)
    output = Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    status = output / "status.json"
    _write_status(status, {"status": "in_progress"})
    target = None
    try:
        implementation = _implementation_provenance()
        manifest_a_hash, manifest_b_hash = _file_sha256(manifest_a), _file_sha256(manifest_b)
        snapshot_a, snapshot_b = output / "source_chico_sur.sqlite3", output / "source_chico_norte.sqlite3"
        _snapshot_database(source_a, snapshot_a)
        _snapshot_database(source_b, snapshot_b)
        _require_no_wal(snapshot_a, "Chico Sur snapshot")
        _require_no_wal(snapshot_b, "Chico Norte snapshot")
        snapshot_hashes = (_file_sha256(snapshot_a), _file_sha256(snapshot_b))
        pair = verify_source_pair(snapshot_a, manifest_a, snapshot_b, manifest_b)
        if (pair["chico_sur"]["network_sha256"], pair["chico_norte"]["network_sha256"]) != expected_network_sha256s:
            raise ValueError("verified source network hashes do not match the explicit selection")
        a_features = verify_supervised_features(snapshot_a, manifest_a)
        b_features = verify_label_free_features(snapshot_b, manifest_b)
        b_targets = extract_label_free_targets(snapshot_b, manifest_b, flood_threshold_m3=1.0)
        if b_targets["labels_sha256"] != expected_b_labels_sha256:
            raise ValueError("Chico Norte label digest differs from the frozen evaluation")
        with closing(sqlite3.connect(snapshot_a.resolve().as_uri() + "?mode=ro", uri=True)) as a_conn, \
             closing(sqlite3.connect(snapshot_b.resolve().as_uri() + "?mode=ro", uri=True)) as b_conn:
            a_conn.execute("BEGIN")
            b_conn.execute("BEGIN")
            identity = plan_joint_identities(a_conn, b_conn)
            a_labels = pd.read_sql_query(
                "SELECT r.run_id,n.node_id,r.inunda,r.vol_inundacion_m3 FROM node_results r "
                "JOIN nodes n ON n.node_pk=r.node_pk AND n.network_id=r.network_id", a_conn)
            final_a = json.loads(manifest_a.read_text(encoding="utf-8"))
            staged_a_path = (manifest_a.parent / final_a["source_manifest_path"]).resolve(strict=True)
            staged_a_path.relative_to(manifest_a.parent)
            staged_a_hash = _file_sha256(staged_a_path)
            if staged_a_hash != final_a["source_manifest_sha256"]:
                raise ValueError("Chico Sur staged source manifest changed before copying")
            staged_a = json.loads(staged_a_path.read_text(encoding="utf-8"))
            staged_b = json.loads(manifest_b.read_text(encoding="utf-8"))
            target = connect_managed_database(output / "joint.sqlite3")
            apply_migrations(target)
            target.execute("BEGIN IMMEDIATE")
            try:
                copied_a = _copy_verified_source(a_conn, target, identity["chico_sur"], staged_a, a_labels)
                copied_b = _copy_verified_source(b_conn, target, identity["chico_norte"], staged_b, b_targets["labels"])
                if copied_a["feature_rows"] != a_features["feature_rows_verified"] or copied_b["feature_rows"] != b_features["feature_rows_verified"]:
                    raise ValueError("copied feature rows differ from verified coverage")
                if target.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise ValueError("joint database contains a foreign key violation")
                target.commit()
            except Exception:
                target.rollback()
                raise
        _require_no_wal(snapshot_a, "Chico Sur snapshot")
        _require_no_wal(snapshot_b, "Chico Norte snapshot")
        if (_file_sha256(snapshot_a), _file_sha256(snapshot_b)) != snapshot_hashes:
            raise ValueError("source snapshot changed while materializing")
        if (_file_sha256(manifest_a), _file_sha256(manifest_b)) != (manifest_a_hash, manifest_b_hash):
            raise ValueError("source manifest changed while materializing")
        if _file_sha256(staged_a_path) != staged_a_hash:
            raise ValueError("Chico Sur staged source manifest changed while materializing")
        if _implementation_provenance() != implementation:
            raise ValueError("implementation files or runtime versions changed while materializing")
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target.close()
        target = None
        package = {"schema_version": "joint_source_materialization_v1",
                   "status": "materialized_not_training_ready", "feature_contract_id": "tabular_v4_21",
                   "flood_threshold_m3": 1.0, "expected_b_labels_sha256": expected_b_labels_sha256,
                   "source_manifests": {"chico_sur": {"path": str(manifest_a), "sha256": manifest_a_hash},
                                        "chico_norte": {"path": str(manifest_b), "sha256": manifest_b_hash}},
                   "source_snapshots": {"chico_sur": {"path": snapshot_a.name, "sha256": snapshot_hashes[0]},
                                        "chico_norte": {"path": snapshot_b.name, "sha256": snapshot_hashes[1]}},
                   "joint_database": {"path": "joint.sqlite3", "sha256": _file_sha256(output / "joint.sqlite3")},
                   "implementation": implementation,
                   "identities": identity, "sources": {"chico_sur": copied_a, "chico_norte": copied_b},
                   "limitations": ["not integrated with multinet preparer", "partitions and training not authorized by this materialization"]}
        manifest = output / "joint_source_manifest.json"
        with manifest.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(package, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
        _write_status(status, {"status": package["status"], "manifest_sha256": _file_sha256(manifest)})
        return package
    except Exception as exc:
        if target is not None:
            target.close()
        _write_status(status, {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)})
        raise


def verify_joint_materialization(directory: str | Path) -> dict:
    """Check a materialized package and its source links, without enabling ML."""
    from contextlib import closing
    root = Path(directory).resolve(strict=True)
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    manifest_path = root / "joint_source_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if (status.get("status") != "materialized_not_training_ready"
            or status.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()):
        raise ValueError("joint manifest status or hash mismatch")
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema_version") != "joint_source_materialization_v1"
            or manifest.get("status") != status["status"] or manifest.get("feature_contract_id") != "tabular_v4_21"
            or manifest.get("flood_threshold_m3") != 1.0):
        raise ValueError("unsupported joint manifest")
    for name in ("chico_sur", "chico_norte"):
        source = manifest["source_manifests"][name]
        if _file_sha256(Path(source["path"]).resolve(strict=True)) != source["sha256"]:
            raise ValueError(f"{name} source manifest hash mismatch")
        snapshot = manifest["source_snapshots"][name]
        snapshot_path = (root / snapshot["path"]).resolve(strict=True)
        snapshot_path.relative_to(root)
        _require_no_wal(snapshot_path, f"{name} snapshot")
        if _file_sha256(snapshot_path) != snapshot["sha256"]:
            raise ValueError(f"{name} snapshot hash mismatch")
    database_record = manifest["joint_database"]
    database = (root / database_record["path"]).resolve(strict=True)
    database.relative_to(root)
    _require_no_wal(database, "joint database")
    if _file_sha256(database) != database_record["sha256"]:
        raise ValueError("joint database hash mismatch")
    expected_runs = set()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok" or conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("joint database integrity check failed")
        for name in ("chico_sur", "chico_norte"):
            source = manifest["sources"][name]
            network = conn.execute("SELECT network_sha256 FROM networks WHERE network_id=?", (source["network_id"],)).fetchone()
            if network is None or network[0] != source["network_sha256"]:
                raise ValueError(f"{name} network identity mismatch")
            eligible_count = 0
            for item in source["runs"]:
                run_id = int(item["run_id"])
                if run_id in expected_runs:
                    raise ValueError("duplicate joint run ID")
                expected_runs.add(run_id)
                quality = conn.execute("SELECT eligible,reason_codes_json FROM run_hydraulic_quality WHERE run_id=?", (run_id,)).fetchone()
                if (quality is None or bool(quality[0]) is not item["eligible"]
                        or json.loads(quality[1]) != item["reason_codes"]):
                    raise ValueError(f"joint QA mismatch for run {run_id}")
                if item["eligible"]:
                    eligible_count += 1
                linked = verify_run_source_artifacts(conn, run_id)
                if (linked["network_id"] != source["network_id"]
                        or linked["source_manifest_sha256"] != item["destination_artifact_manifest_sha256"]):
                    raise ValueError(f"joint source link mismatch for run {run_id}")
            if eligible_count != source["eligible_run_count"] or len(source["runs"]) - eligible_count != source["excluded_run_count"]:
                raise ValueError(f"{name} eligible/excluded counts differ")
        if {row[0] for row in conn.execute("SELECT run_id FROM runs")} != expected_runs:
            raise ValueError("joint run inventory differs from manifest")
        feature_count = conn.execute("SELECT COUNT(*) FROM node_features_v4_21").fetchone()[0]
        label_count = conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0]
        expected_feature_count = sum(manifest["sources"][name]["feature_rows"] for name in ("chico_sur", "chico_norte"))
        expected_label_count = sum(manifest["sources"][name]["target_rows"] for name in ("chico_sur", "chico_norte"))
        if feature_count != expected_feature_count or label_count != expected_label_count or feature_count != label_count:
            raise ValueError("joint feature/label row counts differ from manifest")
        missing = conn.execute("SELECT COUNT(*) FROM node_features_v4_21 f LEFT JOIN node_results r "
                               "ON r.run_id=f.run_id AND r.network_id=f.network_id AND r.node_pk=f.node_pk "
                               "WHERE r.run_id IS NULL").fetchone()[0]
        if missing:
            raise ValueError("joint features and labels have different node keys")
    _require_no_wal(database, "joint database")
    for name in ("chico_sur", "chico_norte"):
        snapshot = manifest["source_snapshots"][name]
        snapshot_path = (root / snapshot["path"]).resolve(strict=True)
        snapshot_path.relative_to(root)
        _require_no_wal(snapshot_path, f"{name} snapshot")
        if _file_sha256(snapshot_path) != snapshot["sha256"]:
            raise ValueError(f"{name} snapshot hash mismatch")
    if _file_sha256(database) != database_record["sha256"] or manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("joint database or manifest changed during verification")
    return {"status": manifest["status"], "feature_rows": feature_count, "target_rows": label_count,
            "runs": len(expected_runs), "scope": "integrity only; not training-ready"}


def _copy_verified_source(source: sqlite3.Connection, target: sqlite3.Connection,
                          identity: dict, manifest: dict, labels) -> dict:
    """Copy a preverified source using caller's remapping, within a savepoint.

    This is an internal building block, not a complete provenance check or a
    publication mechanism. Source files are reverified before registration.
    Caller owns the outer transaction and must verify the resulting package.
    """
    original_network = int(identity["source_network_id"])
    new_network = int(identity["network_id"])
    source_networks = source.execute("SELECT network_id,network_sha256 FROM networks").fetchall()
    if (len(source_networks) != 1 or int(source_networks[0][0]) != original_network
            or source_networks[0][1] != identity["network_sha256"]
            or manifest["network_sha256"] != identity["network_sha256"]):
        raise ValueError("source network identity differs from mapping and manifest")
    nodes = {int(row[0]): str(row[1]) for row in source.execute(
        "SELECT node_pk,node_id FROM nodes WHERE network_id=? ORDER BY node_pk", (original_network,))}
    entries = manifest["run_order"]
    if not nodes or len(set(nodes.values())) != len(nodes):
        raise ValueError("source node inventory is empty or duplicated")
    if len(entries) != len(identity["runs"]) or {int(e["run_id"]) for e in entries} != set(identity["runs"]):
        raise ValueError("source run mapping differs from verified manifest")
    for entry in entries:
        row = source.execute("SELECT manifest_json,policy_json,policy_sha256,created_at_utc FROM run_hydraulic_quality WHERE run_id=?", (entry["run_id"],)).fetchone()
        if row is None:
            raise ValueError("source run has no hydraulic QA")
        quality = json.loads(row[0])
        detailed = {**quality, "policy": json.loads(row[1]), "policy_sha256": row[2], "created_at_utc": row[3]}
        if (quality.get("eligible") is not entry["eligible"] or quality.get("reason_codes") != entry["reason_codes"]
                or detailed != entry["hydraulic_quality"]):
            raise ValueError(f"source QA differs from manifest for run {entry['run_id']}")
    eligible = {int(e["run_id"]) for e in entries if e["eligible"] is True}
    if not eligible:
        raise ValueError("source has no QA-eligible runs")
    if any(source.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
           for table in ("links", "scenario_inflows", "node_features")):
        raise ValueError("source contains unsupported extra tables; refusing to discard data")
    expected = {(run_id, node) for run_id in eligible for node in nodes.values()}
    if not {"run_id", "node_id", "inunda", "vol_inundacion_m3"}.issubset(labels.columns):
        raise ValueError("target labels have an unsupported schema")
    targets = {}
    for row in labels.itertuples(index=False):
        key = (int(row.run_id), str(row.node_id))
        volume, category = float(row.vol_inundacion_m3), float(row.inunda)
        if key in targets or not math.isfinite(volume) or volume < 0 or category not in (0, 1) or category != int(volume >= 1.0):
            raise ValueError("target labels have duplicate keys or invalid flood values")
        targets[key] = (int(category), volume)
    if set(targets) != expected:
        raise ValueError("target labels do not cover exactly the eligible run/node inventory")
    extras = {}
    for old_run, node_id, category, volume, peak, duration, depth, ratio in source.execute(
        "SELECT r.run_id,n.node_id,r.inunda,r.vol_inundacion_m3,r.peak_flooding_lps,"
        "r.flooding_duration_min,r.max_depth_m,r.max_depth_ratio FROM node_results r "
        "JOIN nodes n ON n.node_pk=r.node_pk AND n.network_id=r.network_id"):
        key = (int(old_run), str(node_id))
        if key in extras or key not in targets or targets[key] != (int(category), float(volume)):
            raise ValueError("stored source targets differ from verified labels")
        extras[key] = (peak, duration, depth, ratio)
    source_target_count = source.execute("SELECT COUNT(*) FROM node_results").fetchone()[0]
    if source_target_count != len(extras) or (extras and set(extras) != expected):
        raise ValueError("stored source targets have incomplete or unexpected coverage")

    def copy_table(table, pk=None, mappings=None):
        columns = [row[1] for row in source.execute(f"PRAGMA table_info({table})")]
        target_columns = [row[1] for row in target.execute(f"PRAGMA table_info({table})")]
        if columns != target_columns:
            raise ValueError(f"source and destination schemas differ for {table}")
        insert = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})"
        for row in source.execute(f"SELECT {','.join(columns)} FROM {table}"):
            values = dict(zip(columns, row))
            if "network_id" in values:
                if int(values["network_id"]) != original_network:
                    raise ValueError(f"{table} contains another network")
                values["network_id"] = new_network
            for column, lookup in (mappings or {}).items():
                values[column] = lookup[int(values[column])]
            target.execute(insert, [values[column] for column in columns])

    target.execute("SAVEPOINT copy_joint_source")
    try:
        copy_table("networks")
        copy_table("nodes", mappings={"node_pk": identity["nodes"]})
        copy_table("scenarios", mappings={"scenario_id": identity["scenarios"]})
        copy_table("runs", mappings={"run_id": identity["runs"], "scenario_id": identity["scenarios"]})
        copy_table("run_hydraulic_quality", mappings={"run_id": identity["runs"]})
        feature_keys = set()
        columns = [row[1] for row in source.execute("PRAGMA table_info(node_features_v4_21)")]
        if columns != [row[1] for row in target.execute("PRAGMA table_info(node_features_v4_21)")]:
            raise ValueError("source and destination feature schemas differ")
        insert = f"INSERT INTO node_features_v4_21 ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})"
        for row in source.execute(f"SELECT {','.join(columns)} FROM node_features_v4_21"):
            values = dict(zip(columns, row))
            old_run, old_node = int(values["run_id"]), int(values["node_pk"])
            if int(values["network_id"]) != original_network or old_node not in nodes:
                raise ValueError("feature row has invalid source network/node")
            key = (old_run, nodes[old_node])
            if key in feature_keys or key not in expected or values["feature_contract_id"] != "tabular_v4_21":
                raise ValueError("feature rows do not match eligible inventory or contract")
            feature_keys.add(key)
            values.update(run_id=identity["runs"][old_run], network_id=new_network,
                          node_pk=identity["nodes"][old_node])
            target.execute(insert, [values[column] for column in columns])
        if feature_keys != expected:
            raise ValueError("feature coverage differs from eligible inventory")
        node_by_name = {node: old_pk for old_pk, node in nodes.items()}
        target.executemany(
            "INSERT INTO node_results (run_id,network_id,node_pk,inunda,vol_inundacion_m3,"
            "peak_flooding_lps,flooding_duration_min,max_depth_m,max_depth_ratio) VALUES (?,?,?,?,?,?,?,?,?)",
            ((identity["runs"][run], new_network, identity["nodes"][node_by_name[node]], *values,
              *extras.get((run, node), (None, None, None, None))) for (run, node), values in targets.items()))
        bindings = []
        for entry in entries:
            old_run = int(entry["run_id"])
            before = verify_run_source_artifacts(source, old_run)
            if before["source_manifest_sha256"] != entry["source_manifest_sha256"]:
                raise ValueError(f"source artifact manifest changed for run {old_run}")
            paths = {role: item["resolved_path"] for role, item in before["artifacts"].items()}
            after = register_run_source_artifacts(target, identity["runs"][old_run], paths)
            if any(before["artifacts"][role]["sha256"] != after["artifacts"][role]["sha256"] for role in paths):
                raise ValueError(f"source artifact changed during copy for run {old_run}")
            bindings.append({"source_run_id": old_run, "run_id": identity["runs"][old_run],
                             "source_artifact_manifest_sha256": before["source_manifest_sha256"],
                             "destination_artifact_manifest_sha256": after["source_manifest_sha256"],
                             "eligible": entry["eligible"], "reason_codes": entry["reason_codes"]})
        if target.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("joint database has a foreign key violation")
        target.execute("RELEASE SAVEPOINT copy_joint_source")
        return {"network_sha256": identity["network_sha256"], "source_network_id": original_network,
                "network_id": new_network, "eligible_run_count": len(eligible),
                "excluded_run_count": len(entries) - len(eligible), "target_rows": len(targets),
                "feature_rows": len(feature_keys), "runs": bindings}
    except Exception:
        target.execute("ROLLBACK TO SAVEPOINT copy_joint_source")
        target.execute("RELEASE SAVEPOINT copy_joint_source")
        raise
