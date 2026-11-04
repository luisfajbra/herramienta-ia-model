"""Explicitly materialize a frozen two-network source package; never train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swmm_resilience.database.joint_materialization import materialize_joint_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("sur-db", "sur-manifest", "norte-db", "norte-manifest", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("sur-inp-sha256", "norte-inp-sha256", "norte-labels-sha256"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)
    result = materialize_joint_dataset(args.sur_db, args.sur_manifest, args.norte_db, args.norte_manifest,
                                       args.output, expected_b_labels_sha256=args.norte_labels_sha256,
                                       expected_network_sha256s=(args.sur_inp_sha256, args.norte_inp_sha256))
    print(json.dumps({"status": result["status"], "database": str(args.output / "joint.sqlite3"),
                      "manifest": str(args.output / "joint_source_manifest.json"),
                      "database_sha256": result["joint_database"]["sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
