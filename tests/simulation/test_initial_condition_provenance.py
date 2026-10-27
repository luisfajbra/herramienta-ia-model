from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from swmm_resilience.simulation.batch import _prepare_base_scenario_inp
from swmm_resilience.simulation.initial_conditions import (
    build_initial_condition_manifest,
)
from swmm_resilience.simulation.runner import run_simulation_simple


_NETWORK = """\
[OPTIONS]
FLOW_UNITS LPS
FLOW_ROUTING DYNWAVE
START_DATE 01/01/2025
START_TIME 00:00:00
END_DATE 01/01/2025
END_TIME 00:10:00

[JUNCTIONS]
J1 10.0 2.0 0.25 0.0 0.0
J2 9.0 2.0 0.0 0.0 0.0

[STORAGE]
S1 9.0 2.0 0.75 FUNCTIONAL 1.0 0.5 0.0

[OUTFALLS]
OUT 8.0 FREE NO

[CONDUITS]
C1 J1 J2 100.0 0.013 0.0 0.0 1.5
C2 J2 OUT 50.0 0.013 0.0 0.0 2.5

[INFLOWS]
J1 FLOW TS FLOW 1 2.0 5.0

[TIMESERIES]
TS 0:00 10
TS 0:10 0

[XSECTIONS]
C1 CIRCULAR 0.5 0 0 0
C2 CIRCULAR 0.5 0 0 0
"""


def test_initial_condition_manifest_hashes_link_and_node_state_without_editing_inp(tmp_path):
    inp_path = tmp_path / "initial_state.inp"
    inp_path.write_text(_NETWORK, encoding="utf-8")
    original = inp_path.read_bytes()

    manifest = build_initial_condition_manifest(inp_path)

    assert manifest["policy_id"] == "source_inp_initial_state_v1"
    assert manifest["source_inp_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["state_sha256"] == hashlib.sha256(
        manifest["state_canonical_json"].encode("utf-8")
    ).hexdigest()
    assert manifest["conduit_initial_flows"] == [
        {"link_id": "C1", "flow_initial": 1.5},
        {"link_id": "C2", "flow_initial": 2.5},
    ]
    assert manifest["node_initial_depths"] == [
        {"node_id": "J1", "node_type": "junction", "depth_initial": 0.25},
        {"node_id": "J2", "node_type": "junction", "depth_initial": 0.0},
        {"node_id": "S1", "node_type": "storage", "depth_initial": 0.75},
    ]
    assert inp_path.read_bytes() == original


def test_initial_condition_state_hash_changes_when_one_init_flow_changes(tmp_path):
    first = tmp_path / "first.inp"
    second = tmp_path / "second.inp"
    first.write_text(_NETWORK, encoding="utf-8")
    second.write_text(_NETWORK.replace("C1 J1 J2 100.0 0.013 0.0 0.0 1.5", "C1 J1 J2 100.0 0.013 0.0 0.0 3.0"), encoding="utf-8")

    first_manifest = build_initial_condition_manifest(first)
    second_manifest = build_initial_condition_manifest(second)

    assert first_manifest["state_sha256"] != second_manifest["state_sha256"]
    assert first_manifest["source_inp_sha256"] != second_manifest["source_inp_sha256"]


def test_base_scenario_writes_initial_condition_manifest_from_unchanged_source(tmp_path):
    inp_path = tmp_path / "source.inp"
    inp_path.write_text(_NETWORK, encoding="utf-8")
    original = inp_path.read_bytes()
    config = SimpleNamespace(
        network=SimpleNamespace(inp_path=inp_path),
        validation=SimpleNamespace(inflow_closure_tolerance_lps=0.01),
    )

    scenario_inp, _closure = _prepare_base_scenario_inp(
        config, 1.0, tmp_path / "runs", drain_down_hours=1.0
    )

    manifest_path = scenario_inp.parent / f"{scenario_inp.stem}.initial_condition.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy_id"] == "source_inp_initial_state_v1"
    assert manifest["source_inp_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["conduit_initial_flows"][0] == {
        "link_id": "C1", "flow_initial": 1.5,
    }
    assert inp_path.read_bytes() == original


def test_simple_run_saves_initial_state_manifest_without_modifying_source(
    monkeypatch, tmp_path,
):
    import pyswmm

    inp_path = tmp_path / "source.inp"
    inp_path.write_text(_NETWORK, encoding="utf-8")
    original = inp_path.read_bytes()

    class _Simulation:
        def __init__(self, scenario_path):
            self.scenario_path = Path(scenario_path)

        def __enter__(self):
            self.scenario_path.with_suffix(".rpt").touch()
            return iter(())

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(pyswmm, "Simulation", _Simulation)
    run_dir = tmp_path / "simple_run"

    rpt_path = run_simulation_simple(inp_path, 1.0, run_dir)

    manifest = json.loads(
        (run_dir / "factor_1.0000.initial_condition.json").read_text(encoding="utf-8")
    )
    assert manifest["source_inp_sha256"] == hashlib.sha256(original).hexdigest()
    assert rpt_path.exists()
    assert inp_path.read_bytes() == original
