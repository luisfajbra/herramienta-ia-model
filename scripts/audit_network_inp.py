"""Write a read-only structural inventory for a SWMM INP file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swmm_resilience.validation.network_admission import inspect_network_inp
from swmm_resilience.validation.network_policy import validate_network_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="SWMM .inp file to inspect")
    parser.add_argument("--output", required=True, type=Path, help="destination JSON profile")
    parser.add_argument("--policy", type=Path, help="explicit hydraulic-domain policy JSON")
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output or source.parent == output.parent or source.parent in output.parents:
        raise SystemExit("Output profile must be outside the source INP directory.")

    try:
        profile = inspect_network_inp(source)
        policy = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else {}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Could not read input or policy: {exc}", file=sys.stderr)
        return 3
    issues = validate_network_profile(profile, policy)
    policy_incomplete = bool(issues) and all(
        issue["code"] in {"missing_policy", "missing_policy_value", "invalid_policy_value"}
        for issue in issues
    )
    if not issues:
        status = "supported_by_declared_policy"
    elif policy_incomplete:
        status = "policy_incomplete"
    else:
        status = "rejected_by_declared_policy"
    report = {
        "profile": profile,
        "policy_issues": issues,
        "admission_status": status,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"Could not write report: {exc}", file=sys.stderr)
        return 3
    print(f"INP: {source}")
    print(f"SHA-256: {profile['source']['sha256']}")
    print(
        "Network facts: "
        f"{profile['topology']['node_count']} nodes, "
        f"{profile['topology']['link_count']} links, "
        f"{len(profile['outfalls'])} outfalls, "
        f"{profile['time_series']['count']} time series, "
        f"{len(profile['unresolved_references'])} unresolved references"
    )
    print(f"Policy result: {report['admission_status']} ({len(issues)} issue(s))")
    print(f"Profile: {output}")
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
