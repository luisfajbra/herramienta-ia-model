"""Read-only validation of a Chico Norte synthetic event selection."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

APPROVED_NETWORK_SHA256 = "1c46db327638e023f86f7f0061774abe708f01b8df625f6b6c7f58af36a1ace1"


def _exact_fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} fields must be exactly {sorted(expected)}")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _source(value, base: Path, label: str) -> tuple[Path, bytes, str]:
    _exact_fields(value, ("path", "sha256"), label)
    if not isinstance(value["path"], str) or not value["path"].strip():
        raise ValueError(f"{label} path must be a nonempty string")
    digest = value["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} sha256 must be 64 lowercase hex digits")
    path = (base / value["path"]).resolve(strict=True)
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != digest:
        raise ValueError(f"{label} hash mismatch: {path}")
    return path, contents, digest


def _shape(contents: bytes) -> list[tuple[float, float]]:
    reader = csv.DictReader(io.StringIO(contents.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != ["time_h", "q_norm"]:
        raise ValueError("hydrograph CSV must contain exactly time_h,q_norm")
    try:
        rows = []
        for row in reader:
            if None in row or set(row) != {"time_h", "q_norm"}:
                raise ValueError("hydrograph row has unexpected columns")
            rows.append((float(row["time_h"]), float(row["q_norm"])))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid hydrograph row: {exc}") from exc
    if len(rows) < 3 or any(not math.isfinite(t) or not math.isfinite(q) for t, q in rows):
        raise ValueError("hydrograph needs at least three finite points")
    times, flows = zip(*rows)
    if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("hydrograph times must start at zero and strictly increase")
    if any(abs(t * 60 - round(t * 60)) > 1e-7 for t in times):
        raise ValueError("hydrograph times must be exact to the minute")
    if any(q < 0 or q > 1 for q in flows) or abs(max(flows) - 1.0) > 1e-9:
        raise ValueError("hydrograph flows must be within [0,1] and have a unit peak")
    if abs(flows[0]) > 1e-12 or abs(flows[-1]) > 1e-12:
        raise ValueError("hydrograph must start and end at zero")
    return rows


def inspect_plan(protocol_path: Path | str) -> dict:
    path = Path(protocol_path).resolve(strict=True)
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("protocol must be valid UTF-8 JSON") from exc
    _exact_fields(document, ("schema_version", "network_inp", "hydrographs", "factors", "input_rule"), "protocol")
    if document["schema_version"] != "chico_norte_event_selection_v1":
        raise ValueError("unsupported protocol version")
    if document["input_rule"] != "nodal_peaks_from_inp":
        raise ValueError("unsupported input_rule")
    network_path, _, network_hash = _source(document["network_inp"], path.parent, "network")
    if network_hash != APPROVED_NETWORK_SHA256:
        raise ValueError("network is not the approved Chico Norte INP")

    factors = document["factors"]
    if not isinstance(factors, list) or not factors:
        raise ValueError("factors must be a nonempty list")
    checked = []
    for factor in factors:
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor):
            raise ValueError("factor must be a finite number")
        if not 0.2 <= factor <= 5.0 or abs(factor * 1000 - round(factor * 1000)) > 1e-8:
            raise ValueError("factor must be in [0.2, 5.0] with at most three decimals")
        checked.append(round(float(factor), 3))
    if len(set(checked)) != len(checked):
        raise ValueError("factors must be unique")

    if not isinstance(document["hydrographs"], list) or not document["hydrographs"]:
        raise ValueError("hydrographs must be a nonempty list")
    hydrographs, ids = [], set()
    for item in document["hydrographs"]:
        _exact_fields(item, ("id", "path", "sha256"), "hydrograph")
        ident = item["id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
            raise ValueError("hydrograph id must contain only letters, digits, underscores or hyphens")
        if ident in ids:
            raise ValueError("hydrograph ids must be unique")
        ids.add(ident)
        source, contents, digest = _source({"path": item["path"], "sha256": item["sha256"]}, path.parent, "hydrograph")
        points = _shape(contents)
        hydrographs.append({"id": ident, "path": str(source), "sha256": digest, "points": points})
    return {"schema_version": document["schema_version"], "protocol_path": str(path),
            "protocol_sha256": hashlib.sha256(raw).hexdigest(),
            "network_inp": {"path": str(network_path), "sha256": network_hash},
            "hydrographs": hydrographs, "factors": checked,
            "scenario_count": len(hydrographs) * len(checked), "input_rule": document["input_rule"],
            "scope": "synthetic normalized nodal inflows; not rainfall or observed hydrographs"}


def prepare_plan(protocol_path: Path | str, output: Path | str) -> dict:
    """Materialize selected inputs only; no simulations, labels or database writes."""
    from scripts import run_chico_norte_novel_hydrograph_eval as backend

    selection = inspect_plan(protocol_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    network_hash = hashlib.sha256(backend.NETWORK_INP.read_bytes()).hexdigest()
    if network_hash != selection["network_inp"]["sha256"]:
        raise ValueError("project network differs from selected Chico Norte network")
    config = (backend.REPO_ROOT / "config.yaml").resolve(strict=True)
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    shapes = {item["id"]: item["points"] for item in selection["hydrographs"]}
    paths = {item["id"]: Path(item["path"]) for item in selection["hydrographs"]}
    plan, _, _, nodes, records = backend._build_event_inputs(output, config, shapes, paths, selection["factors"])
    if inspect_plan(protocol_path) != selection:
        raise ValueError("selection sources changed during event preparation")
    if hashlib.sha256(config.read_bytes()).hexdigest() != config_hash:
        raise ValueError("configuration changed during event preparation")
    if hashlib.sha256(backend.NETWORK_INP.read_bytes()).hexdigest() != network_hash:
        raise ValueError("project network changed during event preparation")
    manifest = {"schema_version": "chico_norte_prepared_events_v1", "status": "prepared_not_simulated",
                "selection": selection, "config": {"path": str(config), "sha256": config_hash},
                "effective_network": {"path": str(backend.NETWORK_INP.resolve()), "sha256": network_hash},
                "positive_peak_nodes": nodes, "scenario_plan": plan, "scenarios": records,
                "limitation": "Input preparation only; no immutable simulation protocol or QA yet."}
    manifest_path = output / "prepared_events.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def freeze_plan(protocol_path: Path | str, output: Path | str) -> dict:
    """Prepare and anchor a pre-simulation protocol, without running SWMM."""
    from swmm_resilience.config import load_config
    from swmm_resilience.database.b_transfer_protocol import freeze_b_transfer_protocol
    from swmm_resilience.simulation.frozen_pair_runner import (
        _git_commit, _policy, _runtime_context, build_protocol,
    )
    from scripts import run_chico_norte_novel_hydrograph_eval as backend

    root = Path(output).resolve()
    anchor = root.parent / (root.name + "_protocol_anchor.sqlite3")
    if anchor.exists() or anchor.is_symlink():
        raise FileExistsError(f"refusing to reuse an existing protocol anchor: {anchor}")
    prepared = prepare_plan(protocol_path, root)
    selection = prepared["selection"]
    if inspect_plan(protocol_path) != selection:
        raise ValueError("selection sources changed before freezing")
    config_path = Path(prepared["config"]["path"])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != prepared["config"]["sha256"]:
        raise ValueError("config changed before freezing")
    shapes = {item["id"]: Path(item["path"]) for item in selection["hydrographs"]}
    config = load_config(str(config_path))
    context = _runtime_context(config_path, shapes)
    if context["config_sha256"] != prepared["config"]["sha256"]:
        raise ValueError("config changed while recording the protocol")
    for item in selection["hydrographs"]:
        if context["shape_sources"][item["id"]]["sha256"] != item["sha256"]:
            raise ValueError("hydrograph changed while recording the protocol")
    records = prepared["scenarios"]
    context.update({"network_role": "B", "event_time_policy": backend.TIME_POLICY,
                    "positive_peak_event_nodes": prepared["positive_peak_nodes"], "base_event_nodes": [],
                    "planned_event_sources": {item["scenario_key"]: {"path": item["event_csv"],
                                              "sha256": item["event_csv_sha256"]} for item in records},
                    "selection_protocol": {"path": selection["protocol_path"], "sha256": selection["protocol_sha256"]},
                    "prepared_events_sha256": hashlib.sha256((root / "prepared_events.json").read_bytes()).hexdigest()})
    for record in context["planned_event_sources"].values():
        if hashlib.sha256((root / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("prepared event changed before freezing")
    protocol = build_protocol("B", backend.NETWORK_INP, prepared["scenario_plan"], _policy(config))
    receipt = freeze_b_transfer_protocol(root, protocol, anchor_db_path=anchor,
                                         git_commit=_git_commit(), execution_context=context)
    with (root / "anchor_receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"status": "frozen_before_simulation", "experiment_root": str(root),
            "protocol": str(root / "protocol.json"), "anchor_db": str(anchor),
            "scenario_count": len(records), "simulation_run": False}


def run_frozen_plan(experiment_root: Path | str) -> dict:
    """Run SWMM only for an intact anchored selection, preserving every QA result."""
    from swmm_resilience.database.b_transfer_protocol import verify_b_transfer_protocol_anchor
    from swmm_resilience.simulation import frozen_pair_runner as runner

    root = Path(experiment_root).resolve(strict=True)
    for name in ("batch_progress.json", "runs", "frozen_b_manifest.json", "experiment_source_manifest.json"):
        if (root / name).exists() or (root / name).is_symlink():
            raise FileExistsError(f"refusing to overwrite previous execution: {root / name}")
    protocol_path, receipt_path = root / "protocol.json", root / "anchor_receipt.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    anchor = root.parent / (root.name + "_protocol_anchor.sqlite3")
    verify_b_transfer_protocol_anchor(anchor, protocol_path, receipt)
    context = receipt["execution_context"]
    selection_path = Path(context["selection_protocol"]["path"])
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != context["selection_protocol"]["sha256"]:
        raise ValueError("selected protocol changed after freezing")
    selection = inspect_plan(selection_path)
    prepared_path = root / "prepared_events.json"
    if hashlib.sha256(prepared_path.read_bytes()).hexdigest() != context["prepared_events_sha256"]:
        raise ValueError("prepared events manifest changed after freezing")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared["selection"] != selection:
        # JSON serializes tuples as arrays; compare canonical JSON representations.
        if json.dumps(prepared["selection"], sort_keys=True) != json.dumps(selection, sort_keys=True):
            raise ValueError("prepared selection differs from frozen sources")
    for record in context["planned_event_sources"].values():
        path = (root / record["path"]).resolve(strict=True)
        path.relative_to(root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"event changed after freezing: {path}")
    shape_paths = {item["id"]: Path(item["path"]) for item in selection["hydrographs"]}
    current = runner._runtime_context(Path(context["config_path"]), shape_paths)
    for field in ("pyswmm_version", "swmm_api_version", "swmm_engine_version"):
        if current[field] != context[field]:
            raise ValueError(f"runtime changed after freezing: {field}")
    events = {scenario["scenario_key"]: (scenario["shape_id"], scenario["factor_mult"])
              for scenario in protocol["scenarios"]}
    return runner.run_frozen_experiment("B", Path(selection["network_inp"]["path"]), root, anchor,
                                        protocol, receipt, events, time_policy=context["event_time_policy"])


def score_frozen_plan(experiment_root: Path | str) -> dict:
    """Evaluate both frozen candidates against an already completed SWMM batch."""
    from scripts import run_chico_norte_novel_hydrograph_eval as backend
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations
    from swmm_resilience.database.label_free_b_staging import (
        register_and_predict_staged_transfer, stage_frozen_b_label_free,
    )
    from swmm_resilience.ml.bench.frozen_transfer import score_transfer

    root = Path(experiment_root).resolve(strict=True)
    source_manifest = root / "frozen_b_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"completed SWMM manifest missing: {source_manifest}")
    for name in ("selected_staging.sqlite3", "selected_predictions", "selected_scoring_status.json"):
        if (root / name).exists() or (root / name).is_symlink():
            raise FileExistsError(f"refusing to overwrite previous evaluation: {root / name}")
    registration_path = backend.SOURCE_REGISTRATION
    seal = json.loads(registration_path.with_name("registration_seal.json").read_text(encoding="utf-8"))
    if hashlib.sha256(registration_path.read_bytes()).hexdigest() != seal["sha256"]:
        raise ValueError("domain policy registration hash mismatch")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("b_inp_sha256") != APPROVED_NETWORK_SHA256:
        raise ValueError("domain policy belongs to a different network")
    anchor = root.parent / (root.name + "_protocol_anchor.sqlite3")
    db_path = root / "selected_staging.sqlite3"
    status = root / "selected_scoring_status.json"
    def record(value):
        status.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    record({"status": "running"})
    conn = None
    try:
        conn = connect_managed_database(db_path)
        apply_migrations(conn)
        staged = stage_frozen_b_label_free(conn, root, feature_contract_id="tabular_v4_21",
                                           protocol_anchor_db_path=anchor)
        reports = {}
        for family, candidate in backend.MODEL_CANDIDATES.items():
            transfer_root = root / "selected_predictions" / family
            transfer_root.mkdir(parents=True, exist_ok=False)
            transfer = register_and_predict_staged_transfer(candidate, registration["domain_policy"],
                                                             registration["volume_bands_m3"], transfer_root,
                                                             conn, staged)
            report_path = Path(score_transfer(transfer, conn=conn, staged=staged))
            reports[family] = {"path": str(report_path), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                               "data": json.loads(report_path.read_text(encoding="utf-8"))}
        if conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] != 0:
            raise RuntimeError("evaluation persisted nodal labels unexpectedly")
        eligible = set(reports["mlp"]["data"]["eligible_run_ids"])
        if eligible != set(reports["xgboost"]["data"]["eligible_run_ids"]):
            raise ValueError("models evaluated different eligible runs")
        payload = {"status": "complete", "eligible_scenarios": len(eligible),
                   "scores": {family: {"path": item["path"], "sha256": item["sha256"],
                                       "overall": item["data"]["overall"]} for family, item in reports.items()},
                   "node_results_rows": 0,
                   "scope": "frozen models evaluated against SWMM; no retraining"}
        record(payload)
        return payload
    except Exception as exc:
        record({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if conn is not None:
            conn.close()


def render_selected_figures(experiment_root: Path | str) -> dict:
    """Render only verified eligible scenarios using the approved map style."""
    from contextlib import closing
    import sqlite3
    import numpy as np
    import matplotlib.pyplot as plt
    from scripts import run_chico_norte_novel_hydrograph_eval as backend
    from scripts import build_chico_norte_presentation_figures as figures

    root = Path(experiment_root).resolve(strict=True)
    status_path = root / "selected_scoring_status.json"
    if not status_path.is_file():
        raise FileNotFoundError("completed selected-model scores are required")
    output = root / "selected_figures"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite figures: {output}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("selected-model scoring is not complete")
    source_manifest = root / "experiment_source_manifest.json"
    manifest_sha = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    transfer_dirs, reports = {}, {}
    for family in ("xgboost", "mlp"):
        record = status["scores"][family]
        report_path = Path(record["path"]).resolve(strict=True)
        report_path.relative_to((root / "selected_predictions" / family).resolve())
        if hashlib.sha256(report_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"{family} score hash mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        directory = report_path.parent
        registration = directory / "registration.json"
        prediction = directory / "predictions.csv"
        inputs = directory / "inputs.csv"
        seal = json.loads((directory / "prediction_seal.json").read_text(encoding="utf-8"))
        registration_seal = json.loads((directory / "registration_seal.json").read_text(encoding="utf-8"))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        if (report["b_source_manifest_sha256"] != manifest_sha
                or sha(registration) != report["registration_sha256"]
                or sha(registration) != registration_seal["sha256"]
                or sha(registration) != seal["registration_sha256"]
                or sha(prediction) != report["prediction_sha256"]
                or sha(prediction) != seal["predictions_sha256"]
                or sha(inputs) != report["input_sha256"]
                or sha(inputs) != seal["inputs_sha256"]):
            raise ValueError(f"{family} figures source seals do not match")
        transfer_dirs[family], reports[family] = directory, report
    prepared = json.loads((root / "prepared_events.json").read_text(encoding="utf-8"))
    selection = inspect_plan(prepared["selection"]["protocol_path"])
    if json.dumps(prepared["selection"], sort_keys=True) != json.dumps(selection, sort_keys=True):
        raise ValueError("selected hydrograph sources changed")
    db = root / "selected_staging.sqlite3"
    with closing(sqlite3.connect(db.resolve(strict=True).as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        joined, metadata = backend._load_scored_observations(
            root, {"manifest_path": source_manifest, "conn": conn}, transfer_dirs, reports)
    if joined.empty:
        raise ValueError("no eligible data for figures")
    output.mkdir(exist_ok=False)
    old = (figures.OUTPUT, figures.SHAPE_LABELS)
    selected_ids = {str(item["id"]): item for item in selection["hydrographs"]}
    labels = {ident: ident.replace("_", " ") for ident in selected_ids}
    eligible = [item for item in metadata.values() if item["eligible"]]
    skipped = []
    try:
        figures.OUTPUT, figures.SHAPE_LABELS = output, labels
        for ident, item in selected_ids.items():
            figures.draw_hydrograph(Path(item["path"]), ident)
        maps = []
        for item in eligible:
            for key, label in (("swmm", "SWMM"), ("mlp", "MLP"), ("xgboost", "XGBoost")):
                row = {"scenario_key": item["scenario_key"], "shape_id": item["shape_id"],
                       "factor_mult": item["factor_mult"], "model_key": key, "model_name": label}
                filename = f"mapa_{item['shape_id']}_f{float(item['factor_mult']):.3f}_{key}.png"
                maps.append(figures.draw_network_map(joined, row, backend.NETWORK_INP, output / filename))
        if float(np.ptp(joined["vol_inundacion_m3"].to_numpy(dtype=float))) == 0.0:
            skipped.append("paridad: volumen SWMM constante; R² no está definido")
        else:
            for family in ("mlp", "xgboost"):
                figures.draw_parity(joined, family)
        for ident in sorted(set(joined["shape_id"])):
            group = joined[joined["shape_id"] == ident].groupby("factor_mult", as_index=False).agg(
                swmm=("vol_inundacion_m3", "sum"), mlp=("mlp_volume_m3", "sum"), xgb=("xgb_volume_m3", "sum"))
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for column, label, color in (("swmm", "SWMM", "#333333"), ("mlp", "MLP", "#E87525"), ("xgb", "XGBoost", "#4386C6")):
                ax.plot(group.factor_mult, group[column], marker="o", label=label, color=color)
            ax.set(title=labels[ident], xlabel="Factor multiplicador", ylabel="Volumen de inundación (m³)")
            ax.grid(alpha=0.22)
            ax.legend(frameon=False)
            fig.savefig(output / f"volumen_factor_{ident}.png", dpi=180, facecolor="white")
            plt.close(fig)
        if json.dumps(prepared["selection"], sort_keys=True) != json.dumps(inspect_plan(prepared["selection"]["protocol_path"]), sort_keys=True):
            raise ValueError("hydrograph sources changed while rendering")
        payload = {"status": "complete", "eligible_scenarios": len(eligible),
                   "excluded_scenarios": [item["scenario_key"] for item in metadata.values() if not item["eligible"]],
                   "maps": maps, "skipped": skipped,
                   "score_status_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
                   "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.png"))}}
        with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return payload
    finally:
        figures.OUTPUT, figures.SHAPE_LABELS = old
