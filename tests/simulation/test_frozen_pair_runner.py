from pathlib import Path
import gc
import shutil
from uuid import uuid4

import pytest

from swmm_resilience.simulation.frozen_pair_runner import (
    event_stats,
    expected_scenario_plan,
    write_event_csv,
    inspect_pair,
    prepare_pair,
    drain_down_schedule,
    canonicalize_event_times,
)


def test_event_stats_match_first_sorted_node_feature_convention():
    series = {"Z": [(0.0, 1.0), (2.0, 5.0)], "A": [(0.0, 8.0), (2.0, 3.0)]}
    assert event_stats(series) == (2.0, 0.0)


def test_write_event_csv_materializes_node_time_lps_and_roundtrip():
    out = Path(__file__).parent
    key = f"evt_{uuid4().hex}"
    path = write_event_csv(key, {"N1": [(0.0, 10.0), (1.5, 20.1234567)]}, out,
                           time_policy="require_exact_minute")
    try:
        assert path.read_text(encoding="utf-8").splitlines() == [
            "node_id,time,value_lps", "N1,0:00,10.000000", "N1,1:30,20.123457",
        ]
        from swmm_resilience.validation.hydrograph_csv import load_scenario
        loaded = load_scenario(path, {"N1"})
        assert loaded.node_series["N1"] == [(0.0, 10.0), (1.5, 20.123457)]
    finally:
        path.unlink(missing_ok=True)


def test_event_time_policy_is_explicit_and_detects_minute_rounding():
    values = {"N1": [(0.0, 1), (0.083, 2)]}
    with pytest.raises(ValueError, match="not exact to the minute"):
        canonicalize_event_times(values, time_policy="require_exact_minute")
    rounded = canonicalize_event_times(values, time_policy="round_to_minute")
    assert rounded["N1"][-1][0] == pytest.approx(5.0 / 60.0)


def test_plan_has_25_base_plus_39_shapes_for_a_and_only_shapes_for_b():
    shapes = {f"shape_{i:02d}": [(0.0, 0.0), (1.0, 1.0)] for i in range(39)}
    factors = [round(0.2 * i, 1) for i in range(1, 26)]
    a = expected_scenario_plan("A", shapes, factors)
    b = expected_scenario_plan("B", shapes, factors)
    assert len(a) == 1000
    assert sum(item["shape_id"] == "base" for item in a) == 25
    assert len(b) == 975
    assert all(item["shape_id"] != "base" for item in b)
    assert len({item["scenario_key"] for item in a + b}) == 1975



def test_read_only_preflight_checks_both_project_networks_and_source_catalog():
    report = inspect_pair(Path("config.yaml"))
    assert report["shape_count"] == 39
    assert report["scenario_counts"] == {"A": 1000, "B": 975}
    assert {network["role"] for network in report["networks"]} == {"A", "B"}
    assert all(network["positive_peak_flow_nodes"] > 0 for network in report["networks"])
    assert all(not network["negative_peak_flow_nodes"] for network in report["networks"])
    assert report["time_grid_diagnostics"]["rounded_shape_timestamp_collisions"] == 0
    assert report["time_grid_diagnostics"]["max_rounding_shift_minutes"] == 0.5


def test_invalid_time_policy_fails_before_creating_any_output_root():
    output_root = Path(__file__).parent / f"uncreated-invalid-time-policy-{uuid4().hex}"
    try:
        with pytest.raises(ValueError, match="not exact to the minute"):
            prepare_pair(Path("config.yaml"), output_root, output_root / "anchors.sqlite3",
                         execute=False, freeze_only=False,
                         event_time_policy="require_exact_minute")
        assert not output_root.exists()
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_frozen_drainage_schedule_keeps_base_plus_two_retries():
    assert drain_down_schedule({"drain_down_hours": 6, "drain_down_extension_hours": 2,
                                "max_drain_retries": 2}) == (6.0, 8.0, 10.0)
from pathlib import Path
import hashlib
import json
import sqlite3
from uuid import uuid4

from swmm_resilience.database.b_transfer_protocol import freeze_b_transfer_protocol
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.simulation.frozen_pair_runner import (
    build_protocol,
    run_frozen_experiment,
    write_event_csv,
)


def test_mocked_frozen_execution_records_retries_without_calling_swmm(monkeypatch):
    from swmm_resilience.simulation import frozen_pair_runner as runner
    from swmm_resilience.simulation import timeseries_scenario

    root = Path(__file__).parent
    suffix = uuid4().hex
    inp = Path("data/networks/chico_norte_93_qx1/SWMM - Chico Norte 93% - Qx1.00.inp").resolve()
    anchor_db = Path("tests") / f"anchor-{suffix}.sqlite3"
    key = f"B:mock_{suffix}:f1.000"
    cleanup = [root / "protocol.json", root / "anchor_receipt.json",
               root / "batch_progress.json", root / "batch_progress.json.tmp",
               root / "frozen_b_manifest.json", anchor_db]
    static = extract_static_features(inp)
    expected = set(static.loc[static.base_inflow_lps.astype(float) > 0, "node_id"].astype(str))
    series = {node: [(0.0, 1.0), (0.5, 2.0), (1.0, 0.0)] for node in expected}
    event_path = write_event_csv(key, series, root, time_policy="require_exact_minute")
    cleanup.append(event_path)
    event_hash = hashlib.sha256(event_path.read_bytes()).hexdigest()
    context = {
        "python_version": "test", "pyswmm_version": "test", "swmm_api_version": "test",
        "swmm_engine_version": "test", "config_path": str(Path("config.yaml").resolve()),
        "config_sha256": hashlib.sha256(Path("config.yaml").read_bytes()).hexdigest(),
        "shape_sources": {}, "network_role": "B", "event_time_policy": "require_exact_minute",
        "positive_peak_event_nodes": sorted(expected), "base_event_nodes": [],
        "planned_event_sources": {key: {"path": event_path.name, "sha256": event_hash}},
    }
    policy = {
        "max_continuity_error_pct": 5.0,
        "continuity_limit_rationale": "provisional test policy",
        "drain_down_hours": 6.0, "drain_down_extension_hours": 2.0,
        "max_drain_retries": 2, "tail_window_hours": 1.0, "min_report_steps": 2,
        "inflow_tolerance_lps": 0.01, "flooding_tolerance_lps": 0.01,
        "storage_change_tolerance_m3": 0.01, "residual_flow_tolerance_lps": 0.01,
    }
    plan = [{"scenario_key": key, "shape_id": "mock", "factor_mult": 1.0,
             "duracion_horas": 1.0, "tiempo_al_pico_h": 0.5}]
    protocol = build_protocol("B", inp, plan, policy)
    commit = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt = freeze_b_transfer_protocol(root, protocol, anchor_db_path=anchor_db,
                                         git_commit=commit, execution_context=context)
    cleanup.append(root / "protocol.json")
    (root / "anchor_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    cleanup.append(root / "anchor_receipt.json")
    frozen_protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))

    calls = []
    def fake_write(source, scenario, out_dir, drain_down_hours):
        path = root / f"effective-{len(calls)}.inp"
        path.write_bytes(inp.read_bytes())
        cleanup.append(path)
        return path

    def fake_simulate(effective):
        calls.append(effective)
        rpt, out = effective.with_suffix(".rpt"), effective.with_suffix(".out")
        rpt.write_text("synthetic RPT", encoding="utf-8")
        out.write_bytes(b"synthetic OUT")
        cleanup.extend([rpt, out])
        return rpt, out

    def fake_qa(*, rpt_path, drain_down_hours, **kwargs):
        return {"continuity_error_pct": 0.0, "finite_coverage_ok": True,
                "drainage_state": "drained" if drain_down_hours >= 8 else "not_drained",
                "flooding_report": {"parser": "test", "parser_version": "1",
                                    "report_sha256": hashlib.sha256(rpt_path.read_bytes()).hexdigest()}}

    monkeypatch.setattr(timeseries_scenario, "write_scenario_inp", fake_write)
    monkeypatch.setattr(runner, "_run_swmm", fake_simulate)
    monkeypatch.setattr(Path, "mkdir", lambda self, *args, **kwargs: None)
    monkeypatch.setattr("swmm_resilience.database.label_free_b_staging._derive_hydraulic_quality", fake_qa)
    try:
        result = run_frozen_experiment(
            "B", inp, root, anchor_db, frozen_protocol, receipt, {key: ("mock", 1.0)},
            shapes={}, time_policy="require_exact_minute", simulate=fake_simulate,
        )
        assert result["status"] == "complete"
        assert len(calls) == 2
        progress = json.loads(Path(result["batch_progress_path"]).read_text(encoding="utf-8"))
        attempts = progress["attempt_history"][key]
        assert [entry["drain_down_hours"] for entry in attempts] == [6.0, 8.0]
        assert [entry["state"] for entry in attempts] == ["not_drained", "drained"]
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        assert set(manifest["runs"][0]["artifacts"]) == {
            "source_network_inp", "effective_inp", "event_source", "swmm_rpt", "swmm_out",
        }
        assert manifest["runs"][0]["artifacts"]["event_source"]["sha256"] == event_hash
    finally:
        gc.collect()
        for path in reversed(cleanup):
            path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(anchor_db) + suffix).unlink(missing_ok=True)
