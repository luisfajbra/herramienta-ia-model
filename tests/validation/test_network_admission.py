from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts import audit_network_inp
from swmm_resilience.validation.network_admission import inspect_network_inp


_FIXTURE = """\
[TITLE]
Admission inventory fixture

[OPTIONS]
FLOW_UNITS LPS
INFILTRATION HORTON
FLOW_ROUTING DYNWAVE
LINK_OFFSETS DEPTH
START_DATE 01/01/2025
START_TIME 00:00:00
REPORT_START_DATE 01/01/2025
REPORT_START_TIME 00:00:00
END_DATE 01/01/2025
END_TIME 00:15:00
REPORT_STEP 00:05:00
WET_STEP 00:05:00
DRY_STEP 00:05:00
ROUTING_STEP 00:00:30

[JUNCTIONS]
;;Name  Elevation  MaxDepth  InitDepth  SurDepth  Aponded
J1 2.0 2.0 0.0 0.0
J2 1.0 2.0 0.0 0.0

[STORAGE]
;;Name Elevation MaxDepth InitDepth Shape Curve A1 A2 A0 SurDepth Fevap
S1 0.0 3.0 0.0 FUNCTIONAL 1.0 0.5 0.0

[OUTFALLS]
;;Name Elevation Type StageData Gated
OUT1 -1.0 FIXED 0.25 NO

[CONDUITS]
;;Name FromNode ToNode Length Roughness InOffset OutOffset
C1 J1 J2 100 0.013 0.0 0.0
C1_PAR J1 J2 120 0.014 0.0 0.0
C2 J2 S1 80 0.013 0.0 0.0
C3 J2 J1 90 0.013 0.0 0.0

[PUMPS]
;;Name FromNode ToNode Curve Status Startup Shutoff
P1 S1 OUT1 PUMP_CURVE ON 0.0 0.0

[CURVES]
;;Name Type X-Value Y-Value
PUMP_CURVE PUMP 0.0 0.0
PUMP_CURVE 1.0 5.0

[XSECTIONS]
;;Link Shape Geom1 Geom2 Geom3 Geom4
C1 CIRCULAR 0.5 0 0 0
C1_PAR CIRCULAR 0.4 0 0 0
C2 CIRCULAR 0.3 0 0 0
C3 CIRCULAR 0.3 0 0 0

[INFLOWS]
;;Node Constituent TimeSeries Type Mfactor Sfactor Baseline Pattern
J1 FLOW EVENT_J1 FLOW 1.0 1.0 2.5

[TIMESERIES]
;;Name Date Time Value
EVENT_J1 0:00 1.0
EVENT_J1 0:05 2.0
EVENT_J1 0:10 1.0
EVENT_J1 0:15 0.0
"""


_CLI_POLICY = {
    "policy_version": "test_v1",
    "allowed_flow_units": ["LPS"],
    "allowed_length_units": ["m"],
    "supported_node_types": ["junction", "storage", "outfall"],
    "supported_link_types": ["conduit", "pump"],
    "supported_cross_section_shapes": ["CIRCULAR"],
    "supported_inflow_constituents": ["FLOW"],
    "allow_parallel_links": True,
    "supported_outfall_boundary_types": ["FIXED"],
    "supported_structure_sections": ["STORAGE", "PUMPS"],
    "supported_additional_flow_input_sections": [],
    "allowed_offset_modes": ["DEPTH"],
    "allow_persistent_inflows": True,
    "allow_negative_flow_samples": True,
    "allow_non_finite_flow_samples": True,
    "allow_inflow_patterns": False,
    "allow_rainfall_model": False,
    "allow_missing_geometry": True,
    "feature_contract_version": "tabular_v4_21",
}


def test_inventory_is_deterministic_and_does_not_modify_source(tmp_path):
    inp_path = tmp_path / "small_network.inp"
    inp_path.write_text(_FIXTURE, encoding="utf-8")
    original = inp_path.read_bytes()
    original_mtime = inp_path.stat().st_mtime_ns

    first = inspect_network_inp(inp_path)
    second = inspect_network_inp(inp_path)

    assert first == second
    assert json.dumps(first, sort_keys=True, allow_nan=False)
    assert first["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert first["sections"]["JUNCTIONS"] == 2
    assert first["sections"]["STORAGE"] == 1
    assert first["sections"]["OUTFALLS"] == 1
    assert first["sections"]["CONDUITS"] == 4
    assert first["sections"]["PUMPS"] == 1
    assert first["options"]["FLOW_UNITS"] == "LPS"
    assert {node["node_type"] for node in first["nodes"]} == {
        "junction",
        "storage",
        "outfall",
    }
    assert first["link_offset_mode"] == "DEPTH"
    assert first["hydrology"] == {
        "rain_gage_count": 0,
        "subcatchment_count": 0,
        "has_rainfall_model": False,
    }
    assert first["additional_flow_input_sections"] == {}
    assert first["topology"]["parallel_link_groups"] == [
        {"from_node": "J1", "to_node": "J2", "link_ids": ["C1", "C1_PAR"]}
    ]
    assert first["topology"]["is_directed_acyclic"] is False
    assert first["outfalls"] == [
        {
            "node_id": "OUT1",
            "elevation": -1.0,
            "boundary_type": "FIXED",
            "stage_data": 0.25,
            "has_flap_gate": False,
            "route_to": None,
        }
    ]
    assert first["cross_sections"]["shape_counts"] == {"CIRCULAR": 4}
    assert first["cross_sections"]["barrel_counts"] == {"1": 4}
    assert first["links"][0]["offset_upstream"] == 0.0
    assert first["inflows"]["records"] == [
        {
            "node_id": "J1",
            "constituent": "FLOW",
            "time_series": "EVENT_J1",
            "kind": "FLOW",
            "mass_unit_factor": 1.0,
            "scale_factor": 1.0,
            "baseline": 2.5,
            "baseline_defaulted_before_pattern": 2.5,
            "pattern": None,
        }
    ]
    assert first["time_series"]["records"] == [
        {
            "name": "EVENT_J1",
            "point_count": 4,
            "start_hour": 0.0,
            "end_hour": 0.25,
            "minimum_value": 0.0,
            "maximum_value": 2.0,
            "initial_value": 1.0,
            "final_value": 0.0,
            "nonzero_sample_count": 3,
            "negative_sample_count": 0,
            "non_finite_sample_count": 0,
            "uniform_step_hours": 1.0 / 12.0,
        }
    ]
    assert first["unresolved_references"] == []
    assert inp_path.read_bytes() == original
    assert inp_path.stat().st_mtime_ns == original_mtime


def test_initial_state_hash_changes_when_conduit_init_flow_changes(tmp_path):
    baseline_path = tmp_path / "baseline.inp"
    changed_path = tmp_path / "changed.inp"
    baseline_path.write_text(_FIXTURE, encoding="utf-8")
    changed_inp = _FIXTURE.replace(
        "C1 J1 J2 100 0.013 0.0 0.0\n",
        "C1 J1 J2 100 0.013 0.0 0.0 17.0\n",
    )
    changed_path.write_text(changed_inp, encoding="utf-8")
    original = changed_path.read_bytes()
    original_mtime = changed_path.stat().st_mtime_ns

    baseline = inspect_network_inp(baseline_path)
    changed = inspect_network_inp(changed_path)

    assert baseline["initial_conditions"]["policy_id"] == "as_declared_in_source_inp"
    assert baseline["initial_conditions"]["state_sha256"] != changed["initial_conditions"]["state_sha256"]
    assert changed_path.read_bytes() == original
    assert changed_path.stat().st_mtime_ns == original_mtime


def test_inventory_reports_unresolved_node_and_timeseries_references(tmp_path):
    inp_path = tmp_path / "broken_references.inp"
    inp_path.write_text(
        _FIXTURE.replace("J1 FLOW EVENT_J1", "UNKNOWN FLOW MISSING_TS")
        .replace("C2 J2 S1", "C2 J2 MISSING_NODE")
        .replace("OUT1 -1.0 FIXED 0.25 NO", "OUT1 -1.0 FIXED 0.25 NO SC1")
        .replace(
            "S1 0.0 3.0 0.0 FUNCTIONAL 1.0 0.5 0.0",
            "S1 0.0 3.0 0.0 TABULAR MISSING_STORAGE_CURVE",
        )
        + "\n[RAINGAGES]\nRG1 INTENSITY 0:05 1.0 TIMESERIES MISSING_RAIN_TS\n"
        + "\n[SUBCATCHMENTS]\nSC1 MISSING_RG MISSING_OUTLET 1.0 50 100 1 0\n",
        encoding="utf-8",
    )

    profile = inspect_network_inp(inp_path)

    assert {
        "code": "missing_link_endpoint",
        "object_id": "C2",
        "field": "to_node",
        "reference": "MISSING_NODE",
    } in profile["unresolved_references"]
    assert {
        "code": "missing_inflow_node",
        "object_id": "UNKNOWN:FLOW",
        "field": "node_id",
        "reference": "UNKNOWN",
    } in profile["unresolved_references"]
    assert {
        "code": "missing_inflow_timeseries",
        "object_id": "UNKNOWN:FLOW",
        "field": "time_series",
        "reference": "MISSING_TS",
    } in profile["unresolved_references"]
    assert {
        "code": "missing_storage_curve",
        "object_id": "S1",
        "field": "curve_name",
        "reference": "MISSING_STORAGE_CURVE",
    } in profile["unresolved_references"]
    assert {
        "code": "missing_raingage_timeseries",
        "object_id": "RG1",
        "field": "timeseries",
        "reference": "MISSING_RAIN_TS",
    } in profile["unresolved_references"]
    assert {
        "code": "missing_subcatchment_raingage",
        "object_id": "SC1",
        "field": "rain_gage",
        "reference": "MISSING_RG",
    } in profile["unresolved_references"]
    assert {
        "code": "missing_subcatchment_outlet",
        "object_id": "SC1",
        "field": "outlet",
        "reference": "MISSING_OUTLET",
    } in profile["unresolved_references"]
    assert profile["outfalls"][0]["route_to"] == "SC1"
    assert not any(
        issue["code"] == "missing_outfall_route_subcatchment"
        for issue in profile["unresolved_references"]
    )


def test_inventory_reports_missing_outfall_stage_reference(tmp_path):
    inp_path = tmp_path / "outfall_stage_reference.inp"
    inp_path.write_text(
        _FIXTURE.replace(
            "OUT1 -1.0 FIXED 0.25 NO",
            "OUT1 -1.0 TIMESERIES MISSING_STAGE_TS NO",
        ),
        encoding="utf-8",
    )

    profile = inspect_network_inp(inp_path)

    assert {
        "code": "missing_outfall_stage_timeseries",
        "object_id": "OUT1",
        "field": "stage_data",
        "reference": "MISSING_STAGE_TS",
    } in profile["unresolved_references"]


@pytest.mark.parametrize(
    ("shape_line", "expected_code", "expected_reference"),
    [
        ("C1 CUSTOM 0.5 MISSING_CURVE 0 0", "missing_custom_shape_curve", "MISSING_CURVE"),
        ("C1 IRREGULAR MISSING_TRANSECT 0 0 0", "missing_irregular_transect", "MISSING_TRANSECT"),
        ("C1 STREET MISSING_STREET 0 0 0", "missing_street_geometry", "MISSING_STREET"),
    ],
)
def test_inventory_checks_dependent_cross_section_geometry_references(
    tmp_path, shape_line, expected_code, expected_reference
):
    inp_path = tmp_path / "missing_cross_section_dependency.inp"
    source = _FIXTURE.replace("C1 CIRCULAR 0.5 0 0 0", shape_line)
    inp_path.write_text(source, encoding="utf-8")

    profile = inspect_network_inp(inp_path)

    assert {
        "code": expected_code,
        "object_id": "C1",
        "field": "geometry_reference",
        "reference": expected_reference,
    } in profile["unresolved_references"]


def test_inventory_exposes_additional_persistent_flow_input_sections(tmp_path):
    inp_path = tmp_path / "dwf_network.inp"
    inp_path.write_text(_FIXTURE + "\n[DWF]\nJ1 FLOW 10.0\n", encoding="utf-8")

    profile = inspect_network_inp(inp_path)

    assert profile["additional_flow_input_sections"] == {"DWF": ["J1:FLOW"]}


def test_inventory_is_json_safe_for_non_finite_timeseries_values(tmp_path):
    inp_path = tmp_path / "non_finite_series.inp"
    inp_path.write_text(_FIXTURE.replace("EVENT_J1 0:15 0.0", "EVENT_J1 0:15 NaN"), encoding="utf-8")

    profile = inspect_network_inp(inp_path)

    assert json.dumps(profile, allow_nan=False)
    series = profile["time_series"]["records"][0]
    assert series["non_finite_sample_count"] == 1
    assert series["final_value"] is None


def test_inventory_reports_missing_tidal_outfall_curve(tmp_path):
    inp_path = tmp_path / "outfall_curve_reference.inp"
    inp_path.write_text(
        _FIXTURE.replace(
            "OUT1 -1.0 FIXED 0.25 NO",
            "OUT1 -1.0 TIDAL MISSING_TIDE_CURVE NO",
        ),
        encoding="utf-8",
    )

    profile = inspect_network_inp(inp_path)

    assert {
        "code": "missing_outfall_tidal_curve",
        "object_id": "OUT1",
        "field": "stage_data",
        "reference": "MISSING_TIDE_CURVE",
    } in profile["unresolved_references"]


def test_cli_writes_profile_outside_source_directory(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "profiles"
    source_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    inp_path = source_dir / "network.inp"
    inp_path.write_text(_FIXTURE, encoding="utf-8")
    output_path = output_dir / "profile.json"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_CLI_POLICY), encoding="utf-8")
    original = inp_path.read_bytes()

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_network_inp.py",
            str(inp_path),
            "--policy",
            str(policy_path),
            "--output",
            str(output_path),
        ],
    )

    assert audit_network_inp.main() == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["profile"] == inspect_network_inp(inp_path)
    assert written["policy_issues"] == []
    assert written["admission_status"] == "supported_by_declared_policy"
    assert "Network facts:" in capsys.readouterr().out
    assert inp_path.read_bytes() == original


def test_cli_rejects_output_inside_source_directory(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir(exist_ok=True)
    inp_path = source_dir / "network.inp"
    inp_path.write_text(_FIXTURE, encoding="utf-8")
    output_path = source_dir / "profile.json"
    monkeypatch.setattr(
        "sys.argv",
        ["audit_network_inp.py", str(inp_path), "--output", str(output_path)],
    )

    with pytest.raises(SystemExit, match="outside the source INP directory"):
        audit_network_inp.main()
    assert not output_path.exists()


def test_cli_without_policy_fails_closed(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "profiles"
    source_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    inp_path = source_dir / "network.inp"
    inp_path.write_text(_FIXTURE, encoding="utf-8")
    output_path = output_dir / "profile.json"
    monkeypatch.setattr(
        "sys.argv",
        ["audit_network_inp.py", str(inp_path), "--output", str(output_path)],
    )

    assert audit_network_inp.main() == 2
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["admission_status"] == "policy_incomplete"
    assert any(issue["code"] == "missing_policy_value" for issue in written["policy_issues"])
