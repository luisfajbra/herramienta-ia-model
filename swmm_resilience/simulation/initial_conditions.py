"""Deterministic provenance for the initial hydraulic state in a SWMM INP."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .swmm_api_io import load_inp


INITIAL_CONDITION_POLICY_ID = "source_inp_initial_state_v1"


def _finite_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _records(inp, section_name: str):
    if section_name not in inp:
        return ()
    section = inp[section_name]
    return section.items() if hasattr(section, "items") else ()


def build_initial_condition_manifest(inp_path: Path | str) -> dict:
    """Describe parsed initial flows/depths without changing or judging them.

    The manifest captures SWMM's parsed defaults as well as explicitly supplied
    values. Its source hash preserves the exact input bytes; its state hash is
    stable over the sorted hydraulic initial-state values and flow units.
    """
    path = Path(inp_path)
    source_bytes = path.read_bytes()
    inp = load_inp(path)

    conduit_initial_flows = []
    for link_id, conduit in _records(inp, "CONDUITS"):
        conduit_initial_flows.append({
            "link_id": str(link_id),
            "flow_initial": _finite_number(getattr(conduit, "flow_initial", None)),
        })
    conduit_initial_flows.sort(key=lambda item: item["link_id"])

    node_initial_depths = []
    for section, node_type in (
        ("JUNCTIONS", "junction"),
        ("STORAGE", "storage"),
        ("DIVIDERS", "divider"),
    ):
        for node_id, node in _records(inp, section):
            node_initial_depths.append({
                "node_id": str(node_id),
                "node_type": node_type,
                "depth_initial": _finite_number(getattr(node, "depth_init", None)),
            })
    node_initial_depths.sort(key=lambda item: (item["node_type"], item["node_id"]))

    state = {
        "policy_id": INITIAL_CONDITION_POLICY_ID,
        "flow_units": str(inp["OPTIONS"].get("FLOW_UNITS", "")) if "OPTIONS" in inp else "",
        "conduit_initial_flows": conduit_initial_flows,
        "node_initial_depths": node_initial_depths,
    }
    state_canonical_json = json.dumps(
        state, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {
        **state,
        "source_inp_filename": path.name,
        "source_inp_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "state_canonical_json": state_canonical_json,
        "state_sha256": hashlib.sha256(state_canonical_json.encode("utf-8")).hexdigest(),
    }


def save_initial_condition_manifest(inp_path: Path | str, output_path: Path | str) -> dict:
    """Write one source-state manifest next to scenario artifacts."""
    manifest = build_initial_condition_manifest(inp_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
