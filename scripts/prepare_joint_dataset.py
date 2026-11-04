"""Prepare a verified joint package with explicit protocols; never train."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swmm_resilience.ml.bench.preprocess import prepare_joint_dataset


def verify_partition_hash(path: Path, expected: str) -> None:
    if (len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)
            or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected):
        raise ValueError("partition hash does not match the explicit approved value")


def development_run_ids(package: Path, partition: Path) -> list[int]:
    """Recompute membership from QA and complete shape identities before preparing."""
    package, partition = Path(package), Path(partition)
    status = json.loads((package / "status.json").read_text(encoding="utf-8"))
    data = json.loads(partition.read_text(encoding="utf-8"))
    if (data.get("status") != "membership_only_not_training_ready"
            or data.get("source_manifest_sha256") != status.get("manifest_sha256")):
        raise ValueError("partition does not match the joint package")
    shapes = data.get("reserved_shapes")
    if not isinstance(shapes, list) or not shapes or len(shapes) != len(set(shapes)):
        raise ValueError("partition has invalid reserved shapes")
    with sqlite3.connect((package / "joint.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("""SELECT r.run_id, s.shape_id, q.eligible FROM runs r
            JOIN scenarios s ON s.scenario_id=r.scenario_id AND s.network_id=r.network_id
            JOIN run_hydraulic_quality q ON q.run_id=r.run_id ORDER BY r.run_id""").fetchall()
        n_runs = db.execute("SELECT count(*) FROM runs").fetchone()[0]
    if len(rows) != n_runs:
        raise ValueError("partition cannot verify all runs")
    groups = {"development": [], "holdout": [], "excluded": []}
    for run, shape, eligible in rows:
        group = "excluded" if not eligible else ("holdout" if shape in shapes else "development")
        groups[group].append(run)
    if data.get("run_ids") != groups or not groups["development"] or not groups["holdout"]:
        raise ValueError("partition run membership disagrees with QA or reserved shapes")
    return groups["development"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", action="append", required=True,
                        choices=("LOSO", "GroupKFold5", "ShapeGroupKFold5", "LeaveOneShapeOut", "LeaveOneNetworkOut"))
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--partition-sha256", required=True)
    parser.add_argument("--max-continuity-pct", type=float, required=True)
    args = parser.parse_args(argv)
    if len(set(args.protocol)) != len(args.protocol):
        parser.error("--protocol must not repeat")
    verify_partition_hash(args.partition, args.partition_sha256)
    run_ids = development_run_ids(args.package, args.partition)
    prepared = prepare_joint_dataset(args.package, protocols=tuple(args.protocol), flood_threshold_m3=1.0,
                                     run_ids=run_ids,
                                     output_dir=args.output, max_continuity_error_pct=args.max_continuity_pct)
    print(json.dumps({"status": "prepared_not_trained", "prep_id": prepared.prep_id,
                      "rows": prepared.quality["n_rows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
