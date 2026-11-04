"""Prepare a verified joint package with explicit protocols; never train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swmm_resilience.ml.bench.preprocess import prepare_joint_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", action="append", required=True,
                        choices=("LOSO", "GroupKFold5", "LeaveOneShapeOut", "LeaveOneNetworkOut"))
    parser.add_argument("--max-continuity-pct", type=float, required=True)
    args = parser.parse_args(argv)
    if len(set(args.protocol)) != len(args.protocol):
        parser.error("--protocol must not repeat")
    prepared = prepare_joint_dataset(args.package, protocols=tuple(args.protocol), flood_threshold_m3=1.0,
                                     output_dir=args.output, max_continuity_error_pct=args.max_continuity_pct)
    print(json.dumps({"status": "prepared_not_trained", "prep_id": prepared.prep_id,
                      "rows": prepared.quality["n_rows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
