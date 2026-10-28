"""Reproducible, pre-registered A/B SWMM rerun runner.

This command creates separate immutable roots and requires an explicit --execute
before starting simulations. Existing outputs are never selected or overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

from ..config import load_config
from ..database.b_transfer_protocol import freeze_b_transfer_protocol
from ..extraction.static_features import extract_static_features
from ..simulation.batch import _base_inflow_scenario
from ..simulation.hydrograph_shapes import apply_shape, load_all_shapes
from ..simulation.swmm_api_io import load_inp
from ..validation.hydrograph_csv import HydrographScenario

FACTOR_GRID = tuple(round(0.2 * i, 1) for i in range(1, 26))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _time_minutes(value: float) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"event time {value!r} must be finite and non-negative")
    return round(numeric * 60)


def canonicalize_event_times(
    node_series: dict[str, list[tuple[float, float]]], *, time_policy: str,
) -> dict[str, list[tuple[float, float]]]:
    """Apply an explicit event time policy; no implicit timestamp conversion."""
    if time_policy not in {"round_to_minute", "require_exact_minute"}:
        raise ValueError("time_policy must be explicit: round_to_minute or require_exact_minute")
    output = {}
    for node_id, series in node_series.items():
        rounded = []
        for time_h, value in series:
            minute = _time_minutes(time_h)
            if time_policy == "require_exact_minute" and abs(float(time_h) * 60.0 - minute) > 1e-7:
                raise ValueError(f"event time {time_h!r} is not exact to the minute")
            rounded.append((minute / 60.0, float(value)))
        if any(right[0] <= left[0] for left, right in zip(rounded, rounded[1:])):
            raise ValueError(f"minute rounding collapses or reorders event times at node {node_id}")
        output[str(node_id)] = rounded
    return output


def canonicalize_shape(shape: list[tuple[float, float]], *, time_policy: str) -> list[tuple[float, float]]:
    return canonicalize_event_times({"shape": shape}, time_policy=time_policy)["shape"]


def prevalidate_event_time_policy(shapes: dict, a_input_obj, *, time_policy: str) -> None:
    """Validate every source timestamp before creating any experiment output."""
    for shape_id, shape in shapes.items():
        canonical = canonicalize_shape(shape, time_policy=time_policy)
        event_stats({"shape": [(time_h, round(value, 6)) for time_h, value in canonical]})
    for factor in FACTOR_GRID:
        base = _base_inflow_scenario(a_input_obj, factor, f"preflight_base_f{factor:.3f}")
        canonical = canonicalize_event_times(base.node_series, time_policy=time_policy)
        rounded = {node: [(time_h, round(value, 6)) for time_h, value in series]
                   for node, series in canonical.items()}
        event_stats(rounded)


def event_stats(node_series: dict[str, list[tuple[float, float]]]) -> tuple[float, float]:
    """Use _make_features convention: last time; first max of first sorted node."""
    if not node_series:
        raise ValueError("event has no node series")
    first_id = sorted(node_series)[0]
    reference = node_series[first_id]
    if len(reference) < 2:
        raise ValueError("event series needs at least two time points")
    duration = float(reference[-1][0])
    peak_time = float(max(reference, key=lambda point: point[1])[0])
    for node_id, series in node_series.items():
        if len(series) != len(reference) or [p[0] for p in series] != [p[0] for p in reference]:
            raise ValueError(f"event node {node_id} does not share the exact time grid")
        if any(not math.isfinite(float(q)) or float(q) < 0 for _, q in series):
            raise ValueError(f"event node {node_id} has invalid negative/non-finite flow")
    return duration, peak_time


def write_event_csv(
    scenario_key: str, node_series: dict[str, list[tuple[float, float]]], out_dir: Path,
    *, time_policy: str,
) -> Path:
    node_series = canonicalize_event_times(node_series, time_policy=time_policy)
    node_series = {node: [(time_h, round(value, 6)) for time_h, value in series]
                   for node, series in node_series.items()}
    event_stats(node_series)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario_key)
    path = out_dir / f"{safe}.csv"
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("node_id", "time", "value_lps"))
        for node_id in sorted(node_series):
            for time_h, value in node_series[node_id]:
                writer.writerow((node_id, f"{_time_minutes(time_h)//60}:{_time_minutes(time_h)%60:02d}", f"{float(value):.6f}"))
    return path


def expected_scenario_plan(network_role: str, shapes: dict, factors=FACTOR_GRID) -> list[dict]:
    if network_role not in {"A", "B"}:
        raise ValueError("network_role must be A or B")
    if tuple(float(f) for f in factors) != FACTOR_GRID:
        raise ValueError("rerun requires the frozen 25 factors 0.2 through 5.0 by 0.2")
    if len(shapes) != 39:
        raise ValueError(f"rerun requires exactly 39 versioned shape CSVs; found {len(shapes)}")
    items = []
    shape_specs = [("base", None)] if network_role == "A" else []
    shape_specs.extend((shape_id, shape) for shape_id, shape in sorted(shapes.items()))
    for shape_id, shape in shape_specs:
        for factor in FACTOR_GRID:
            key = f"{network_role}:{shape_id}:f{factor:.3f}"
            duration, peak_time = (0.0, 0.0) if shape is None else (float(shape[-1][0]), float(max(shape, key=lambda p: p[1])[0]))
            items.append({
                "scenario_key": key, "shape_id": shape_id,
                "factor_mult": factor, "duracion_horas": duration,
                "tiempo_al_pico_h": peak_time,
            })
    expected = 1000 if network_role == "A" else 975
    if len(items) != expected or len({row["scenario_key"] for row in items}) != expected:
        raise ValueError(f"invalid {network_role} scenario plan cardinality/identity")
    return items


def drain_down_schedule(policy: dict) -> tuple[float, ...]:
    base = float(policy["drain_down_hours"])
    increment = float(policy["drain_down_extension_hours"])
    retries = int(policy["max_drain_retries"])
    if not math.isfinite(base) or not math.isfinite(increment) or base <= 0 or increment <= 0 or retries < 0:
        raise ValueError("invalid frozen drain-down retry policy")
    return tuple(base + attempt * increment for attempt in range(retries + 1))


def _policy(config) -> dict:
    validation = config.validation
    return {
        "max_continuity_error_pct": float(validation.max_continuity_error_pct),
        "continuity_limit_rationale": "Provisional project threshold retained for this reproducible rerun; hydraulic justification remains pending.",
        "drain_down_hours": float(validation.drain_down_hours),
        "drain_down_extension_hours": float(validation.drainage_extension_increment_hours),
        "max_drain_retries": int(validation.drainage_extension_max_retries),
        "tail_window_hours": float(validation.drainage_tail_window_hours),
        "min_report_steps": 2,
        "inflow_tolerance_lps": float(validation.drainage_inflow_tolerance_lps),
        "flooding_tolerance_lps": float(validation.drainage_flooding_tolerance_lps),
        "storage_change_tolerance_m3": float(validation.drainage_storage_change_tolerance_m3),
        "residual_flow_tolerance_lps": float(validation.drainage_inflow_tolerance_lps),
    }


def _verify_execution_context(context: dict) -> None:
    config_path = Path(context["config_path"]).resolve(strict=True)
    if _sha(config_path) != context["config_sha256"]:
        raise ValueError("config.yaml changed after the run protocol was anchored")
    for shape_id, item in context["shape_sources"].items():
        path = Path(item["path"]).resolve(strict=True)
        if _sha(path) != item["sha256"]:
            raise ValueError(f"frozen hydrograph shape changed after preflight: {shape_id}")


def _runtime_context(config_path: Path, shape_paths: dict[str, Path]) -> dict:
    from pyswmm import __version__ as pyswmm_version
    from swmm_api import __version__ as swmm_api_version
    from swmm.toolkit import solver
    return {
        "python_version": sys.version,
        "pyswmm_version": str(pyswmm_version),
        "swmm_api_version": str(swmm_api_version),
        "swmm_engine_version": str(solver.swmm_version_info()),
        "config_path": str(config_path.resolve()), "config_sha256": _sha(config_path),
        "shape_sources": {key: {"path": str(path.resolve()), "sha256": _sha(path)} for key, path in sorted(shape_paths.items())},
    }


def _git_commit(*, require_clean: bool = False) -> str:
    if require_clean:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        if status.strip():
            raise ValueError("freeze/execute requires a clean committed Git worktree")
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise ValueError("could not resolve a full current Git commit")
    return value


def _network_events(network_role: str, shapes: dict, factors: tuple[float, ...]):
    """Return only compact identities; materialize each full node event at execution time."""
    events = {}
    if network_role == "A":
        for factor in factors:
            events[f"A:base:f{factor:.3f}"] = ("base", factor)
    for shape_id in sorted(shapes):
        for factor in factors:
            events[f"{network_role}:{shape_id}:f{factor:.3f}"] = (shape_id, factor)
    return events


def build_protocol(
    network_role: str, inp_path: Path, scenarios: list[dict], policy: dict,
) -> dict:
    return {
        "schema_version": "b_transfer_protocol_v2",
        "status": "frozen_before_run",
        "network": {"path": str(inp_path.resolve()), "sha256": _sha(inp_path), "name": network_role},
        "expected_scenario_keys": [item["scenario_key"] for item in scenarios],
        "scenarios": scenarios,
        "hydraulic_policy": policy,
    }


def _run_swmm(inp_path: Path) -> tuple[Path, Path]:
    from pyswmm import Simulation
    with Simulation(str(inp_path)) as simulation:
        for _ in simulation:
            pass
    rpt, out = inp_path.with_suffix(".rpt"), inp_path.with_suffix(".out")
    if not rpt.is_file() or not out.is_file() or rpt.stat().st_size == 0 or out.stat().st_size == 0:
        raise RuntimeError("SWMM completed without non-empty RPT and OUT artifacts")
    return rpt, out


def _artifact(root: Path, role: str, path: Path, *, external=False) -> dict:
    resolved = path.resolve(strict=True)
    if not external:
        resolved.relative_to(root.resolve())
    payload = resolved.read_bytes()
    if not payload:
        raise ValueError(f"artifact {role} is empty: {resolved}")
    return {"path": str(resolved) if external else resolved.relative_to(root.resolve()).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest()}


def run_frozen_experiment(
    network_role: str, inp_path: Path, experiment_root: Path, anchor_db: Path,
    protocol: dict, anchor_receipt: dict, events: dict, *, time_policy: str,
    simulate: Callable[[Path], tuple[Path, Path]] = _run_swmm,
    shapes: dict | None = None,
) -> dict:
    """Run every frozen scenario; retain failed QA and fail closed on SWMM errors."""
    from ..database.label_free_b_staging import _derive_hydraulic_quality, _inp_object_ids
    from ..database.hydraulic_quality import build_hydraulic_quality_decision
    from ..simulation.timeseries_scenario import write_scenario_inp
    from ..simulation.initial_conditions import build_initial_condition_manifest
    from ..validation.hydrograph_csv import load_scenario
    from ..simulation.swmm_api_io import load_inp
    from ..database.b_transfer_protocol import verify_b_transfer_protocol_anchor

    root = experiment_root.resolve(strict=True)
    protocol_path = root / "protocol.json"
    receipt_path = root / "anchor_receipt.json"
    if not protocol_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("frozen pre-run anchor receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("in-memory protocol differs from its frozen bytes")
    verify_b_transfer_protocol_anchor(anchor_db, protocol_path, receipt)
    frozen_network = protocol.get("network", {})
    if _sha(inp_path.resolve(strict=True)) != frozen_network.get("sha256"):
        raise ValueError("source INP changed after the pre-run protocol was anchored")
    source_context = receipt["execution_context"]
    _verify_execution_context(source_context)
    protocol_sha = _sha(protocol_path)
    expected = protocol["expected_scenario_keys"]
    if set(events) != set(expected):
        raise ValueError("materialized event keys differ from the pre-registered protocol")
    planned_event_sources = source_context.get("planned_event_sources")
    if not isinstance(planned_event_sources, dict) or set(planned_event_sources) != set(expected):
        raise ValueError("anchored event-source inventory differs from the frozen scenario keys")
    if source_context.get("event_time_policy") != time_policy:
        raise ValueError("event-time policy differs from the externally anchored choice")
    node_ids, link_ids = _inp_object_ids(inp_path)
    policy = protocol["hydraulic_policy"]
    results, failures, attempt_history = [], [], {}
    progress_path = root / "batch_progress.json"

    def save_progress(status):
        _json_progress = {"schema_version": "frozen_batch_progress_v2", "status": status,
                          "protocol_sha256": protocol_sha, "results": results, "failures": failures,
                          "attempt_history": attempt_history}
        temp = progress_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(_json_progress, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, progress_path)

    save_progress("running")
    positive_nodes = set(source_context["positive_peak_event_nodes"])
    base_nodes = set(source_context.get("base_event_nodes", ()))
    for definition in protocol["scenarios"]:
        key = definition["scenario_key"]
        phase = "event_materialization"
        event_path = None
        attempts = []
        attempt_history[key] = attempts
        try:
            shape_id, factor = events[key]
            source_record = planned_event_sources[key]
            event_path = (root / source_record["path"]).resolve(strict=True)
            event_path.relative_to(root)
            if _sha(event_path) != source_record["sha256"]:
                raise ValueError(f"materialized event CSV changed after protocol freeze: {key}")
            expected_nodes = base_nodes if shape_id == "base" else positive_nodes
            parsed = load_scenario(event_path, expected_nodes)
            duration, peak_time = event_stats(parsed.node_series)
            if (abs(duration - definition["duracion_horas"]) > 1e-6
                    or abs(peak_time - definition["tiempo_al_pico_h"]) > 1e-6):
                raise ValueError(f"frozen duration/peak disagrees with materialized event for {key}")
            run_root = root / "runs" / re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
            final = None
            phase = "simulation"
            for attempt, drain_hours in enumerate(drain_down_schedule(policy)):
                attempt_dir = run_root / f"attempt_{attempt:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                safe_scenario_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
                scenario = HydrographScenario(safe_scenario_id, parsed.node_series, parsed.time_grid_hours, parsed.last_time_hours)
                effective_inp = write_scenario_inp(inp_path, scenario, attempt_dir, drain_down_hours=drain_hours)
                try:
                    rpt, out = simulate(effective_inp)
                except Exception as exc:
                    partial = {"attempt": attempt, "drain_down_hours": drain_hours,
                               "state": "simulation_error", "error": repr(exc),
                               "effective_inp": _artifact(root, "effective_inp", effective_inp)}
                    for role, path in (("swmm_rpt", effective_inp.with_suffix(".rpt")),
                                       ("swmm_out", effective_inp.with_suffix(".out"))):
                        if path.is_file() and path.stat().st_size > 0:
                            partial[role] = _artifact(root, role, path)
                        elif path.exists():
                            partial[f"{role}_empty_path"] = path.relative_to(root).as_posix()
                    attempts.append(partial)
                    raise
                attempt_record = {"attempt": attempt, "drain_down_hours": drain_hours,
                                  "state": "qa_pending",
                                  "effective_inp": _artifact(root, "effective_inp", effective_inp),
                                  "swmm_rpt": _artifact(root, "swmm_rpt", rpt),
                                  "swmm_out": _artifact(root, "swmm_out", out)}
                attempts.append(attempt_record)
                phase = "qa"
                try:
                    qa = _derive_hydraulic_quality(
                        source_inp=inp_path, effective_inp=effective_inp, rpt_path=rpt, out_path=out,
                        expected_node_ids=node_ids, expected_link_ids=link_ids, policy=policy,
                        drain_down_hours=drain_hours,
                    )
                except Exception as exc:
                    attempt_record.update({"state": "qa_error", "error": repr(exc)})
                    raise
                attempt_record.update({"state": qa["drainage_state"],
                                       "continuity_error_pct": qa["continuity_error_pct"],
                                       "finite_coverage_ok": qa["finite_coverage_ok"]})
                phase = "simulation"
                final = (effective_inp, rpt, out, qa, drain_hours)
                if qa["drainage_state"] == "drained":
                    break
            effective_inp, rpt, out, qa, drain_hours = final
            source_initial = build_initial_condition_manifest(inp_path)
            effective_initial = build_initial_condition_manifest(effective_inp)
            initial = {"policy_id": source_initial["policy_id"], "state_sha256": source_initial["state_sha256"]}
            if source_initial["state_sha256"] != effective_initial["state_sha256"]:
                raise ValueError("effective INP changed initial hydraulic state")
            continuity = qa["continuity_error_pct"]
            decision = build_hydraulic_quality_decision(
                continuity_error_pct=continuity, max_continuity_error_pct=policy["max_continuity_error_pct"],
                continuity_limit_rationale=policy["continuity_limit_rationale"],
                finite_coverage_ok=qa["finite_coverage_ok"], drainage_state=qa["drainage_state"],
                initial_condition=initial, flooding_report_manifest=qa["flooding_report"],
            )
            quality = {"continuity_error_pct": continuity,
                       "max_continuity_error_pct": policy["max_continuity_error_pct"],
                       "continuity_limit_rationale": policy["continuity_limit_rationale"],
                       "finite_coverage_ok": qa["finite_coverage_ok"], "drainage_state": qa["drainage_state"],
                       "initial_condition": initial, "flooding_report": qa["flooding_report"]}
            artifacts = {
                "source_network_inp": _artifact(root, "source_network_inp", inp_path, external=True),
                "effective_inp": _artifact(root, "effective_inp", effective_inp),
                "event_source": _artifact(root, "event_source", event_path),
                "swmm_rpt": _artifact(root, "swmm_rpt", rpt),
                "swmm_out": _artifact(root, "swmm_out", out),
            }
            record = {"scenario_key": key, "hydraulic_quality": quality, "artifacts": artifacts}
            results.append(record)
            if not decision["eligible"]:
                failures.append({"scenario_key": key, "kind": "hydraulic_qa_ineligible", "reason_codes": decision["reason_codes"]})
        except Exception as exc:
            failures.append({"scenario_key": key, "kind": "simulation_or_qa_error", "phase": phase, "error": repr(exc)})
            if not attempts or attempts[-1].get("error") != repr(exc):
                failed = {"state": f"{phase}_error", "error": repr(exc)}
                if event_path is not None and event_path.is_file():
                    failed["event_source"] = _artifact(root, "event_source", event_path)
                attempts.append(failed)
        save_progress("running")
    status = "complete" if len(results) == len(expected) and not any(f.get("kind") == "simulation_or_qa_error" for f in failures) else "incomplete"
    save_progress(status)
    final_manifest_path = None
    if status == "complete":
        manifest_name = "frozen_b_manifest.json" if network_role == "B" else "frozen_a_training_manifest.json"
        final_manifest_path = root / manifest_name
        _json(final_manifest_path, {
            "schema_version": "frozen_b_transfer_manifest_v1" if network_role == "B" else "frozen_a_training_manifest_v1",
            "status": "complete", "protocol_path": "protocol.json", "protocol_sha256": protocol_sha,
            "protocol_anchor": receipt, "runs": results,
        })
    return {"status": status, "expected": len(expected), "completed": len(results), "failures": failures,
            "batch_progress_path": str(progress_path), "manifest_path": str(final_manifest_path) if final_manifest_path else None,
            "protocol_path": str(protocol_path)}


def inspect_pair(config_path: Path) -> dict:
    """Read-only validation of both networks, factor grid and source catalogs."""
    config = load_config(str(config_path))
    if tuple(float(f) for f in config.factors()) != FACTOR_GRID:
        raise ValueError("config.yaml factor grid differs from the approved 0.2-to-5.0/0.2 grid")
    shapes_dir = config.simulation.hydrograph_shapes_dir
    shape_paths = {path.stem: path for path in sorted(shapes_dir.glob("*.csv"))}
    hashes_before_load = {key: _sha(path) for key, path in shape_paths.items()}
    shapes = load_all_shapes(shapes_dir)
    if hashes_before_load != {key: _sha(path) for key, path in shape_paths.items()}:
        raise ValueError("hydrograph shape files changed while loading the frozen source catalog")
    if len(shapes) != 39 or set(shapes) != set(shape_paths):
        raise ValueError(f"expected 39 unique versioned shape CSVs; found {len(shapes)}")
    b_inp = config.network.inp_path.parent.parent / "chico_norte_93_qx1" / "SWMM - Chico Norte 93% - Qx1.00.inp"
    networks = []
    for role, path in (("A", config.network.inp_path), ("B", b_inp)):
        path = path.resolve(strict=True)
        static = extract_static_features(path)
        if static.empty or static["node_id"].astype(str).duplicated().any():
            raise ValueError(f"network {role} has no unique junction inventory")
        positive = static.loc[static["base_inflow_lps"].astype(float) > 0.0, "node_id"].astype(str)
        if positive.empty:
            raise ValueError(f"network {role} has no positive nodal peaks")
        networks.append({"role": role, "inp": str(path), "sha256": _sha(path),
                         "junction_count": int(len(static)), "positive_peak_flow_nodes": int(len(positive)),
                         "zero_peak_flow_nodes": static.loc[static["base_inflow_lps"].astype(float) == 0.0,
                                                               "node_id"].astype(str).tolist(),
                         "negative_peak_flow_nodes": static.loc[static["base_inflow_lps"].astype(float) < 0.0,
                                                                  "node_id"].astype(str).tolist()})
    # Validate that the 39 source curves can be represented on the project's minute grid.
    time_shifts = [abs(time_h * 60.0 - round(time_h * 60.0)) for shape in shapes.values() for time_h, _ in shape]
    # Validate all A-native base event factors without retaining the full run ensemble.
    a_inp = load_inp(config.network.inp_path)
    base_time_shifts = [abs(time_h * 60.0 - round(time_h * 60.0))
                        for factor in FACTOR_GRID
                        for series in _base_inflow_scenario(a_inp, factor, f"preflight_base_f{factor:.3f}").node_series.values()
                        for time_h, _ in series]
    all_shifts = time_shifts + base_time_shifts
    git_status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return {"networks": networks, "shape_count": len(shapes),
            "shape_sources": {key: {"path": str(path.resolve()), "sha256": _sha(path)}
                              for key, path in sorted(shape_paths.items())},
            "factors": list(FACTOR_GRID), "scenario_counts": {"A": 1000, "B": 975},
            "time_grid_diagnostics": {"non_minute_source_points": sum(value > 1e-7 for value in all_shifts),
                                      "max_rounding_shift_minutes": max(all_shifts, default=0.0),
                                      "rounded_shape_timestamp_collisions": sum(
                                          len({round(t * 60) for t, _ in shape}) != len(shape)
                                          for shape in shapes.values())},
            "execution_context": _runtime_context(config_path, shape_paths),
            "git_commit": _git_commit(), "git_worktree_clean": not bool(git_status.strip())}


def prepare_pair(config_path: Path, output_root: Path, anchor_db: Path, *, execute: bool, freeze_only: bool,
                 event_time_policy: str):
    preflight = inspect_pair(config_path)
    config = load_config(str(config_path))
    shapes_dir = config.simulation.hydrograph_shapes_dir
    shape_paths = {path.stem: path for path in sorted(shapes_dir.glob("*.csv"))}
    hashes_before_load = {key: _sha(path) for key, path in shape_paths.items()}
    shapes = load_all_shapes(shapes_dir)
    if hashes_before_load != {key: _sha(path) for key, path in shape_paths.items()}:
        raise ValueError("hydrograph shape files changed while loading the frozen source catalog")
    commit = _git_commit(require_clean=(execute or freeze_only))
    context = _runtime_context(config_path, shape_paths)
    if preflight["shape_sources"] != context["shape_sources"]:
        raise ValueError("hydrograph shape catalog changed after read-only preflight")
    b_preflight = Path(preflight["networks"][1]["inp"])
    for frozen_network, current_path in zip(preflight["networks"], (config.network.inp_path, b_preflight)):
        if _sha(current_path.resolve(strict=True)) != frozen_network["sha256"]:
            raise ValueError(f"network {frozen_network['role']} INP changed after preflight")
    from ..simulation.swmm_api_io import load_inp
    a_input_obj = load_inp(config.network.inp_path)
    prevalidate_event_time_policy(shapes, a_input_obj, time_policy=event_time_policy)
    context["git_commit"] = commit
    context["event_time_policy"] = event_time_policy
    pairs = [("A", config.network.inp_path)]
    # The B input is a project-managed source, not an inferred user Downloads path.
    b_inp = config.network.inp_path.parent.parent / "chico_norte_93_qx1" / "SWMM - Chico Norte 93% - Qx1.00.inp"
    pairs.append(("B", b_inp))
    pair_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pair_root = output_root / pair_id
    pair_root.mkdir(parents=True, exist_ok=False)
    results = []
    for role, inp in pairs:
        inp = inp.resolve(strict=True)
        plan = expected_scenario_plan(role, shapes, FACTOR_GRID)
        events = _network_events(role, shapes, FACTOR_GRID)
        root = pair_root / role
        root.mkdir(exist_ok=False)
        static = extract_static_features(inp)
        peaks = {str(row.node_id): float(row.base_inflow_lps) for row in static.itertuples()
                 if float(row.base_inflow_lps) > 0.0}
        inp_obj = a_input_obj if role == "A" else load_inp(inp)
        base_nodes = sorted(_base_inflow_scenario(inp_obj, 1.0, "base_preflight").node_series) if role == "A" else []
        event_source_context = {}
        for definition in plan:
            key = definition["scenario_key"]
            shape_id, factor = events[key]
            if shape_id == "base":
                node_series = _base_inflow_scenario(inp_obj, factor, key).node_series
            else:
                shape = canonicalize_shape(shapes[shape_id], time_policy=event_time_policy)
                node_series = apply_shape(shape, peaks, factor)
            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
            event_path = write_event_csv(
                key, node_series, root / "runs" / safe_key / "source", time_policy=event_time_policy
            )
            from ..validation.hydrograph_csv import load_scenario
            expected_nodes = set(base_nodes) if shape_id == "base" else set(peaks)
            materialized = load_scenario(event_path, expected_nodes)
            duration, peak_time = event_stats(materialized.node_series)
            definition["duracion_horas"] = duration
            definition["tiempo_al_pico_h"] = peak_time
            event_source_context[key] = {
                "path": event_path.relative_to(root).as_posix(), "sha256": _sha(event_path),
            }
        protocol = build_protocol(role, inp, plan, _policy(config))
        role_context = {**context, "network_role": role, "event_time_policy": event_time_policy,
                        "positive_peak_event_nodes": sorted(peaks), "base_event_nodes": base_nodes,
                        "planned_event_sources": event_source_context}
        results.append({"role": role, "root": str(root), "input_sha256": _sha(inp), "scenario_count": len(plan),
                        "protocol": protocol, "events": events, "anchor_db": anchor_db,
                        "context": role_context, "freeze_only": freeze_only, "execute": execute})
    if not execute and not freeze_only:
        return [{key: value for key, value in item.items() if key not in {"events", "protocol", "context"}} for item in results]
    # Roots and both protocols are frozen before the first simulation starts.
    frozen = []
    for item in results:
        receipt = freeze_b_transfer_protocol(
            item["root"], item["protocol"], anchor_db_path=anchor_db,
            git_commit=commit, execution_context=item["context"],
        )
        _json(Path(item["root"]) / "anchor_receipt.json", receipt)
        item["protocol"] = json.loads((Path(item["root"]) / "protocol.json").read_text(encoding="utf-8"))
        frozen.append((item, receipt))
    if freeze_only:
        return [{"role": item["role"], "root": item["root"], "protocol_sha256": rec["protocol_sha256"]} for item, rec in frozen]
    # Run only after both A and B roots have pre-run anchors.
    for item, receipt in frozen:
        result = run_frozen_experiment(item["role"], Path(item["protocol"]["network"]["path"]), Path(item["root"]), anchor_db,
                                       item["protocol"], receipt, item["events"], shapes=shapes,
                                       time_policy=event_time_policy)
        item["result"] = result
    return [{"role": item["role"], "root": item["root"], **item.get("result", {})} for item, _ in frozen]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/frozen_pair_rerun_20261028"))
    parser.add_argument("--anchor-db", type=Path, default=Path("outputs/experiments/protocol_anchors.sqlite3"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="inspect networks, forms, counts and hashes without writing files or starting simulations")
    mode.add_argument("--freeze-only", action="store_true", help="create/anchor A+B protocols but do not simulate")
    mode.add_argument("--execute", action="store_true", help="freeze both protocols, then run all SWMM scenarios")
    parser.add_argument("--event-time-policy", choices=("round_to_minute", "require_exact_minute"),
                        help="explicit timestamp policy; required for freeze/execute because current source grids are sub-minute")
    args = parser.parse_args(argv)
    if args.dry_run:
        report = inspect_pair(args.config.resolve())
        print(json.dumps({**report, "total_scenarios": 1975,
                          "output_root": str(args.output_root.resolve()),
                          "simulation_started": False}, indent=2, ensure_ascii=False))
        return 0
    if args.event_time_policy is None:
        parser.error("--freeze-only/--execute requires an explicit --event-time-policy; no protocol was frozen")
    result = prepare_pair(args.config.resolve(), args.output_root.resolve(), args.anchor_db.resolve(),
                          execute=args.execute, freeze_only=args.freeze_only,
                          event_time_policy=args.event_time_policy)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if all(row.get("status", "frozen") in {"complete", "frozen"} for row in result) else 2


if __name__ == "__main__":
    raise SystemExit(main())
