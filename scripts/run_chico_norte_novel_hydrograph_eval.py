"""Run a preregistered two-shape Chico Norte transfer evaluation.

The script intentionally uses a tiny, fixed, documented evaluation design. It
does not train, tune, promote, or alter any model, network INP, or QA threshold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd

from swmm_resilience.config import load_config
from swmm_resilience.database.b_transfer_protocol import freeze_b_transfer_protocol
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.label_free_b_staging import (
    register_and_predict_staged_transfer,
    stage_frozen_b_label_free,
)
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.extraction.labels import extract_labels
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.simulation.frozen_pair_runner import (
    _git_commit,
    _json,
    _policy,
    _runtime_context,
    build_protocol,
    event_stats,
    run_frozen_experiment,
    write_event_csv,
)
from swmm_resilience.simulation.hydrograph_shapes import apply_shape, load_all_shapes
from swmm_resilience.simulation.inflow_closure import apply_closure_policy
from swmm_resilience.validation.hydrograph_csv import HydrographScenario, load_scenario
from swmm_resilience.ml.bench.frozen_transfer import score_transfer


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_INP = REPO_ROOT / "data/networks/chico_norte_93_qx1/SWMM - Chico Norte 93% - Qx1.00.inp"
SHAPES_DIR = REPO_ROOT / "data/hydrograph_validation/chico_norte_nuevos_2026_11_02"
FIGURE_DIR = REPO_ROOT / "docs/figures_chico_norte_novel_eval_2026_11_02"
SOURCE_REGISTRATION = REPO_ROOT / "outputs/bench/multinetwork/20261030T005341Z-813323e3ef75/A_to_B_frozen/registration.json"
EXPERIMENTS = REPO_ROOT / "outputs/experiments"
MODEL_CANDIDATES = {
    "xgboost": REPO_ROOT / "outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/A/candidates/xgboost",
    "mlp": REPO_ROOT / "outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z/A/candidates/mlp/ensemble",
}
FACTOR_SEED = 20261102
FACTOR_COUNT = 5
FLOOD_THRESHOLD_M3 = 1.0
VOLUME_BANDS_M3 = [24.0, 108.0, 344.0]
TIME_POLICY = "round_to_minute"
SHAPE_LABELS = {
    "novel_20261102_cascade_peaks": "Pulsos dobles separados",
    "novel_20261102_sustained_late_surge": "Meseta sostenida y pico tardío",
}


def draw_unseen_factors(seed: int, count: int = FACTOR_COUNT) -> list[float]:
    """Draw reproducible off-grid factors inside the training support [0.2, 5.0]."""
    pool = [round(0.3 + 0.2 * index, 1) for index in range(24)]
    if isinstance(count, bool) or count < 1 or count > len(pool):
        raise ValueError("count must be between 1 and the number of unseen factors")
    return sorted(random.Random(seed).sample(pool, count))


def validate_normalized_shape(shape: list[tuple[float, float]]) -> None:
    """Require a minute-grid normalized shape with zero endpoints and unit peak."""
    if len(shape) < 3:
        raise ValueError("a normalized shape needs at least three points")
    times = [float(time_h) for time_h, _ in shape]
    values = [float(q_norm) for _, q_norm in shape]
    if not all(math.isfinite(value) for value in (*times, *values)):
        raise ValueError("shape times and normalized flows must be finite")
    if times[0] != 0.0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("shape times must start at zero and be strictly increasing")
    minute_times = [round(value * 60) for value in times]
    if any(abs(value * 60 - minute) > 1e-7 for value, minute in zip(times, minute_times)):
        raise ValueError("shape times must be exact to the minute")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("normalized flow must be within [0, 1]")
    if abs(max(values) - 1.0) > 1e-9:
        raise ValueError("normalized shape must have a unit peak")
    if abs(values[0]) > 1e-12 or abs(values[-1]) > 1e-12:
        raise ValueError("normalized shape must start and end at zero")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _load_shape_inputs() -> tuple[dict[str, list[tuple[float, float]]], dict[str, Path]]:
    shapes = load_all_shapes(SHAPES_DIR)
    if set(shapes) != set(SHAPE_LABELS):
        raise ValueError(f"expected exactly the two documented new shapes; found {sorted(shapes)}")
    shape_paths = {path.stem: path.resolve(strict=True) for path in SHAPES_DIR.glob("*.csv")}
    if set(shape_paths) != set(shapes):
        raise ValueError("shape source files and loaded shape IDs differ")
    for shape in shapes.values():
        validate_normalized_shape(shape)
    return shapes, shape_paths


def _preflight() -> dict:
    if not NETWORK_INP.is_file():
        raise FileNotFoundError(f"Chico Norte INP missing: {NETWORK_INP}")
    for model_name, candidate_dir in MODEL_CANDIDATES.items():
        if not (candidate_dir / "metadata.json").is_file():
            raise FileNotFoundError(f"{model_name} candidate missing: {candidate_dir}")
    if not SOURCE_REGISTRATION.is_file():
        raise FileNotFoundError(f"existing domain-policy registration missing: {SOURCE_REGISTRATION}")
    registration = json.loads(SOURCE_REGISTRATION.read_text(encoding="utf-8"))
    registration_seal_path = SOURCE_REGISTRATION.with_name("registration_seal.json")
    registration_seal = json.loads(registration_seal_path.read_text(encoding="utf-8"))
    if registration_seal.get("sha256") != _sha(SOURCE_REGISTRATION):
        raise ValueError("the saved Chico Norte domain-policy registration seal is invalid")
    if registration.get("b_inp_sha256") != _sha(NETWORK_INP):
        raise ValueError("the saved domain-policy registration belongs to a different Chico Norte INP")
    if registration.get("domain_policy", {}).get("feature_contract_version") != "tabular_v4_21":
        raise ValueError("the saved Chico Norte admission policy does not match tabular_v4_21")
    shapes, _ = _load_shape_inputs()
    factors = draw_unseen_factors(FACTOR_SEED, FACTOR_COUNT)
    return {
        "network_inp": str(NETWORK_INP),
        "network_sha256": _sha(NETWORK_INP),
        "hydrograph_shape_ids": list(shapes),
        "factors_draw_order": [3.9, 0.7, 2.7, 4.3, 1.3],
        "factors_sorted_for_reporting": factors,
        "factor_seed": FACTOR_SEED,
        "scenario_count": len(shapes) * len(factors),
        "model_candidates": {name: str(path) for name, path in MODEL_CANDIDATES.items()},
        "domain_policy_id": registration.get("domain_policy", {}).get("policy_version"),
        "source_registration_sha256": _sha(SOURCE_REGISTRATION),
    }


def _build_event_inputs(root: Path, config_path: Path, shapes, shape_paths, factors):
    config = load_config(str(config_path))
    static = extract_static_features(NETWORK_INP)
    peaks = {
        str(row.node_id): float(row.base_inflow_lps)
        for row in static.itertuples()
        if float(row.base_inflow_lps) > 0.0
    }
    positive_nodes = sorted(peaks)
    event_root = root / "events"
    event_root.mkdir(parents=True, exist_ok=False)
    plan, events, planned_sources = [], {}, {}
    generation_rows = []
    for shape_id in sorted(shapes):
        shape = shapes[shape_id]
        for factor in factors:
            key = f"B:{shape_id}:f{factor:.3f}"
            node_series = apply_shape(shape, peaks, factor)
            time_grid = [time_h for time_h, _ in shape]
            scenario = HydrographScenario(key, node_series, time_grid, float(time_grid[-1]))
            closed, closure_manifest = apply_closure_policy(
                scenario, config.validation.inflow_closure_tolerance_lps
            )
            event_path = write_event_csv(
                key, closed.node_series, event_root, time_policy=TIME_POLICY
            )
            parsed = load_scenario(event_path, set(positive_nodes))
            duration, peak_time = event_stats(parsed.node_series)
            plan.append({
                "scenario_key": key,
                "shape_id": shape_id,
                "factor_mult": factor,
                "duracion_horas": duration,
                "tiempo_al_pico_h": peak_time,
            })
            events[key] = (shape_id, factor)
            relative_event = event_path.resolve().relative_to(root.resolve()).as_posix()
            planned_sources[key] = {"path": relative_event, "sha256": _sha(event_path)}
            generation_rows.append({
                "scenario_key": key,
                "shape_id": shape_id,
                "factor_mult": factor,
                "event_csv": relative_event,
                "event_csv_sha256": _sha(event_path),
                "closure_policy": closure_manifest,
                "duration_h": duration,
                "time_to_peak_h": peak_time,
                "peak_reference": "preserve_each_node_peak_from_Chico_Norte_INP_then_multiply_by_factor",
            })
    return plan, events, planned_sources, positive_nodes, generation_rows


def _draw_design_figure(figure_dir: Path, shapes, factors: list[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for shape_id, shape in sorted(shapes.items()):
        axes[0].plot([p[0] for p in shape], [p[1] for p in shape], marker="o",
                     linewidth=2, markersize=4, label=SHAPE_LABELS[shape_id])
    axes[0].set_xlabel("Tiempo desde el inicio del evento (h)")
    axes[0].set_ylabel("Caudal normalizado, q/qmáx")
    axes[0].set_ylim(-0.04, 1.08)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title("Hidrogramas sintéticos nuevos")

    trained = [round(0.2 * index, 1) for index in range(1, 26)]
    axes[1].scatter(trained, np.zeros(len(trained)), color="#9aa0a6", s=24,
                    label="Factores usados en entrenamiento (0,2–5,0)")
    axes[1].scatter(factors, np.ones(len(factors)), color="#e87525", s=75, zorder=3,
                    label="Factores aleatorios de esta prueba")
    for factor in factors:
        axes[1].annotate(f"{factor:.1f}", (factor, 1), xytext=(0, 10),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[1].set_xlim(0, 5.2)
    axes[1].set_yticks([0, 1], ["Entrenamiento", "Evaluación"])
    axes[1].set_xlabel("Factor multiplicador nodal")
    axes[1].set_title("Selección reproducible fuera de la grilla")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(figure_dir / "hidrogramas_y_factores.png", dpi=180)
    plt.close(fig)


def _load_scored_observations(root: Path, staged: dict, transfer_dirs: dict[str, Path], score_reports: dict):
    from swmm_resilience.database.run_source_artifacts import verify_run_source_artifacts

    manifest = json.loads(Path(staged["manifest_path"]).read_text(encoding="utf-8"))
    eligible = set(score_reports["xgboost"]["eligible_run_ids"])
    if eligible != set(score_reports["mlp"]["eligible_run_ids"]):
        raise ValueError("XGBoost and MLP were scored on different QA-eligible run IDs")
    prediction_frames = {}
    for family, directory in transfer_dirs.items():
        seal = json.loads((directory / "prediction_seal.json").read_text(encoding="utf-8"))
        if hashlib.sha256((directory / "predictions.csv").read_bytes()).hexdigest() != seal["predictions_sha256"]:
            raise ValueError(f"{family} prediction file hash differs from its seal")
        prediction_frames[family] = pd.read_csv(directory / "predictions.csv", dtype={"node_id": str})
    xgb = prediction_frames["xgboost"]
    mlp = prediction_frames["mlp"]
    join_keys = ["run_id", "node_id"]
    if not xgb[join_keys].equals(mlp[join_keys]):
        raise ValueError("candidate predictions do not share exact run/node keys")
    network_nodes = extract_static_features(NETWORK_INP)["node_id"].astype(str).tolist()
    labels_frames = []
    run_meta = {}
    for entry in manifest["run_order"]:
        run_id = int(entry["run_id"])
        key = entry["scenario_key"]
        run_meta[run_id] = {**entry["scenario"], "scenario_key": key,
                            "eligible": bool(entry["eligible"]),
                            "reason_codes": entry["reason_codes"],
                            "hydraulic_quality": entry["hydraulic_quality"]}
        if run_id not in eligible:
            continue
        artifact = verify_run_source_artifacts(staged["conn"], run_id)["artifacts"]["swmm_rpt"]
        rpt_path = Path(artifact["resolved_path"]).resolve(strict=True)
        if _sha(rpt_path) != artifact["sha256"]:
            raise ValueError(f"SWMM RPT hash changed before figure generation for run {run_id}")
        labels = extract_labels(rpt_path, network_nodes, FLOOD_THRESHOLD_M3)
        labels["run_id"] = run_id
        labels_frames.append(labels[["run_id", "node_id", "inunda", "vol_inundacion_m3"]])
    if not labels_frames:
        raise ValueError("no QA-eligible runs are available for descriptive figures")
    truth = pd.concat(labels_frames, ignore_index=True)
    joined = truth.merge(xgb, on=join_keys, how="inner", validate="one_to_one")
    joined = joined.merge(
        mlp[join_keys + ["y_prob_clf", "y_pred_clf", "volume_routed_m3"]],
        on=join_keys, how="inner", validate="one_to_one", suffixes=("_xgb", "_mlp"),
    )
    if len(joined) != len(xgb) or len(joined) != len(mlp):
        raise ValueError("predictions and RPT-derived truth have incomplete coverage")
    joined = joined.rename(columns={
        "y_prob_clf_xgb": "xgb_probability", "y_pred_clf_xgb": "xgb_class",
        "volume_routed_m3_xgb": "xgb_volume_m3",
        "y_prob_clf_mlp": "mlp_probability", "y_pred_clf_mlp": "mlp_class",
        "volume_routed_m3_mlp": "mlp_volume_m3",
    })
    joined["scenario_key"] = joined["run_id"].map(lambda run_id: run_meta[int(run_id)]["scenario_key"])
    joined["shape_id"] = joined["run_id"].map(lambda run_id: run_meta[int(run_id)]["shape_id"])
    joined["factor_mult"] = joined["run_id"].map(lambda run_id: run_meta[int(run_id)]["factor_mult"])
    return joined, run_meta


def _draw_parity(figure_dir: Path, joined: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 7.3), constrained_layout=True)
    truth = np.log10(1.0 + joined["vol_inundacion_m3"].to_numpy(dtype=float))
    xgb = np.log10(1.0 + joined["xgb_volume_m3"].to_numpy(dtype=float))
    mlp = np.log10(1.0 + joined["mlp_volume_m3"].to_numpy(dtype=float))
    ax.scatter(truth, xgb, s=16, alpha=0.33, color="#4386c6", label="XGBoost")
    ax.scatter(truth, mlp, s=16, alpha=0.33, color="#e87525", label="MLP (5 semillas)")
    max_value = max(float(truth.max()), float(xgb.max()), float(mlp.max()), 0.1)
    ax.plot([0, max_value], [0, max_value], color="#333333", linewidth=1.2, label="Predicción = SWMM")
    ax.set_xlim(0, max_value * 1.02)
    ax.set_ylim(0, max_value * 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Volumen SWMM: log₁₀(1 + m³)")
    ax.set_ylabel("Volumen predicho: log₁₀(1 + m³)")
    ax.set_title("Paridad global en escenarios elegibles")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "paridad_global.png", dpi=180)
    plt.close(fig)


def _draw_volume_curves(figure_dir: Path, joined: pd.DataFrame) -> pd.DataFrame:
    totals = joined.groupby(["shape_id", "factor_mult"], as_index=False).agg(
        swmm_m3=("vol_inundacion_m3", "sum"),
        xgboost_m3=("xgb_volume_m3", "sum"),
        mlp_m3=("mlp_volume_m3", "sum"),
    )
    fig, axes = plt.subplots(1, len(SHAPE_LABELS), figsize=(13, 4.8), constrained_layout=True)
    if len(SHAPE_LABELS) == 1:
        axes = [axes]
    for ax, shape_id in zip(axes, sorted(SHAPE_LABELS)):
        group = totals[totals.shape_id == shape_id].sort_values("factor_mult")
        ax.plot(group.factor_mult, group.swmm_m3, marker="o", label="SWMM", color="#333333")
        ax.plot(group.factor_mult, group.xgboost_m3, marker="o", label="XGBoost", color="#4386c6")
        ax.plot(group.factor_mult, group.mlp_m3, marker="o", label="MLP (5 semillas)", color="#e87525")
        ax.set_title(SHAPE_LABELS[shape_id])
        ax.set_xlabel("Factor multiplicador")
        ax.set_ylabel("Volumen total predicho / SWMM (m³)")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    fig.savefig(figure_dir / "volumen_total_por_factor.png", dpi=180)
    plt.close(fig)
    return totals


def _draw_network_map(figure_dir: Path, shape_id: str, factor: float, rows: pd.DataFrame,
                      static: pd.DataFrame) -> None:
    frame = rows.merge(static[["node_id", "coord_x", "coord_y"]], on="node_id", how="left", validate="one_to_one")
    if frame[["coord_x", "coord_y"]].isna().any().any():
        raise ValueError("network map needs finite node coordinates from the INP")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), constrained_layout=True)
    values = [frame["vol_inundacion_m3"].to_numpy(float),
              frame["xgb_volume_m3"].to_numpy(float),
              frame["mlp_volume_m3"].to_numpy(float)]
    labels = ["SWMM", "XGBoost", "MLP (5 semillas)"]
    vmax = max(float(np.max(value)) for value in values)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=max(vmax, 1.0))
    for idx, (ax, volume, label) in enumerate(zip(axes, values, labels)):
        flooded = volume >= FLOOD_THRESHOLD_M3
        ax.scatter(frame.loc[~flooded, "coord_x"], frame.loc[~flooded, "coord_y"],
                   s=9, color="#d4d7dc", linewidths=0, zorder=1)
        scatter = ax.scatter(frame.loc[flooded, "coord_x"], frame.loc[flooded, "coord_y"],
                             c=volume[flooded], cmap="magma", norm=norm, s=25,
                             linewidths=0.3, edgecolors="white", zorder=2)
        if idx:
            actual = frame["vol_inundacion_m3"].to_numpy(float) >= FLOOD_THRESHOLD_M3
            false_positive = flooded & ~actual
            false_negative = actual & ~flooded
            ax.scatter(frame.loc[false_positive, "coord_x"], frame.loc[false_positive, "coord_y"],
                       s=40, facecolors="none", edgecolors="#7851a9", linewidths=0.9,
                       label="Falsa alarma", zorder=3)
            ax.scatter(frame.loc[false_negative, "coord_x"], frame.loc[false_negative, "coord_y"],
                       s=38, marker="x", color="#d64b42", linewidths=1.0,
                       label="Inundación omitida", zorder=3)
            if false_positive.any() or false_negative.any():
                ax.legend(loc="best", frameon=True, fontsize=8)
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    fig.colorbar(scatter, ax=axes, shrink=0.78, label="Volumen de inundación (m³; escala raíz)")
    fig.suptitle(f"Chico Norte — {SHAPE_LABELS[shape_id]} — factor {factor:.1f}", fontsize=13)
    fig.savefig(figure_dir / f"mapa_{shape_id}_f{factor:.1f}.png", dpi=180)
    plt.close(fig)


def run_evaluation(experiment_root: Path, figure_dir: Path) -> dict:
    config_path = (REPO_ROOT / "config.yaml").resolve(strict=True)
    config = load_config(str(config_path))
    shapes, shape_paths = _load_shape_inputs()
    factors = draw_unseen_factors(FACTOR_SEED, FACTOR_COUNT)
    preflight = _preflight()
    experiment_root = experiment_root.resolve()
    if experiment_root.exists():
        raise FileExistsError(f"refusing to overwrite experiment root: {experiment_root}")
    experiment_root.mkdir(parents=True, exist_ok=False)

    base = REPO_ROOT / "outputs/experiments/chico_norte_novel_hydrograph_eval_20261102"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = experiment_root
    anchor_db = run_root.parent / f"{run_root.name}_protocol_anchor.sqlite3"
    plan, events, planned_sources, positive_nodes, generation_rows = _build_event_inputs(
        run_root, config_path, shapes, shape_paths, factors
    )
    design = {
        "design_version": "chico_norte_novel_hydrograph_eval_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "objective": "evaluate frozen Chico Sur-trained XGBoost and five-seed MLP on new synthetic inflow shapes in Chico Norte against SWMM",
        "network_inp": {"path": str(NETWORK_INP.resolve()), "sha256": _sha(NETWORK_INP)},
        "hydrograph_files": {name: {"path": str(path), "sha256": _sha(path)} for name, path in sorted(shape_paths.items())},
        "hydrograph_labels": SHAPE_LABELS,
        "factor_selection": {
            "algorithm": "Python random.Random(seed).sample without replacement from off-training tenths",
            "seed": FACTOR_SEED,
            "pool": [round(0.3 + 0.2 * index, 1) for index in range(24)],
            "draw_order": [3.9, 0.7, 2.7, 4.3, 1.3],
            "sorted_for_reporting": factors,
            "interpretation": "all factors are inside training support [0.2, 5.0], but absent from the 0.2-spaced training grid; this is interpolation in factor, not factor extrapolation",
        },
        "input_construction": {
            "scope": "nodal FLOW hydrographs, no rainfall-runoff transformation",
            "peak_policy": "preserve each Chico Norte INP node peak and multiply by the selected factor",
            "time_policy": TIME_POLICY,
            "closure_policy": "linear_v1 applied before event CSV freeze; endpoints are zero",
            "scenario_count": len(plan),
            "scenarios": generation_rows,
        },
        "qa_policy": _policy(config),
        "models": {name: {"candidate_dir": str(path.resolve()), "metadata_sha256": _sha(path / "metadata.json"),
                          "classifier_sha256": _sha(path / "classifier.joblib"),
                          "regressor_sha256": _sha(path / "regressor.joblib")} for name, path in MODEL_CANDIDATES.items()},
        "feature_contract": "tabular_v4_21",
        "flood_threshold_m3": FLOOD_THRESHOLD_M3,
        "volume_bands_m3": VOLUME_BANDS_M3,
        "preflight": preflight,
    }
    design_path = run_root / "evaluation_design.json"
    _write_json(design_path, design)

    protocol = build_protocol("B", NETWORK_INP, plan, _policy(config))
    context = _runtime_context(config_path, shape_paths)
    context.update({
        "network_role": "B",
        "event_time_policy": TIME_POLICY,
        "positive_peak_event_nodes": positive_nodes,
        "base_event_nodes": [],
        "planned_event_sources": planned_sources,
        "evaluation_design": {"path": design_path.name, "sha256": _sha(design_path)},
        "factor_random_seed": FACTOR_SEED,
        "working_tree_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True),
    })
    receipt = freeze_b_transfer_protocol(
        run_root, protocol, anchor_db_path=anchor_db, git_commit=_git_commit(),
        execution_context=context,
    )
    _json(run_root / "anchor_receipt.json", receipt)
    frozen_protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
    run_result = run_frozen_experiment(
        "B", NETWORK_INP, run_root, anchor_db, frozen_protocol, receipt, events,
        time_policy=TIME_POLICY, shapes=shapes,
    )
    if run_result["status"] != "complete" or run_result["completed"] != len(plan):
        raise RuntimeError(f"SWMM experiment incomplete; inspect {run_result['batch_progress_path']}")

    db_path = run_root / "novel_hydrographs_staging.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        staged = stage_frozen_b_label_free(
            conn, run_root, feature_contract_id="tabular_v4_21",
            protocol_anchor_db_path=anchor_db,
        )
        staged["conn"] = conn
        source_registration = json.loads(SOURCE_REGISTRATION.read_text(encoding="utf-8"))
        domain_policy = source_registration["domain_policy"]
        bands = source_registration["volume_bands_m3"]
        transfer_dirs, score_reports = {}, {}
        for family, candidate_dir in MODEL_CANDIDATES.items():
            transfer_root = run_root / "predictions" / family
            transfer_root.mkdir(parents=True, exist_ok=False)
            transfer_dir = register_and_predict_staged_transfer(
                candidate_dir, domain_policy, bands, transfer_root, conn, staged
            )
            transfer_dirs[family] = transfer_dir
            score_path = score_transfer(transfer_dir, conn=conn, staged=staged)
            score_reports[family] = json.loads(Path(score_path).read_text(encoding="utf-8"))
        if conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0] != 0:
            raise RuntimeError("verified evaluation unexpectedly persisted node targets to SQLite")
        joined, run_meta = _load_scored_observations(run_root, staged, transfer_dirs, score_reports)
        static = extract_static_features(NETWORK_INP)
        figure_dir = figure_dir.resolve()
        figure_dir.mkdir(parents=True, exist_ok=False)
        _draw_design_figure(figure_dir, shapes, factors)
        _draw_parity(figure_dir, joined)
        totals = _draw_volume_curves(figure_dir, joined)
        target_factor = min(factors, key=lambda value: (abs(value - 2.5), value))
        selected_maps = {}
        for shape_id in sorted(SHAPE_LABELS):
            eligible_meta = [item for item in run_meta.values()
                             if item["shape_id"] == shape_id and item["eligible"]]
            if not eligible_meta:
                selected_maps[shape_id] = None
                continue
            selected = min(eligible_meta, key=lambda item: (abs(float(item["factor_mult"]) - target_factor),
                                                             float(item["factor_mult"])))
            factor = float(selected["factor_mult"])
            scenario_key = selected["scenario_key"]
            map_rows = joined[joined["scenario_key"] == scenario_key]
            _draw_network_map(figure_dir, shape_id, factor, map_rows, static)
            selected_maps[shape_id] = {"scenario_key": scenario_key, "factor_mult": factor,
                                       "selection_rule": "eligible scenario nearest the median sampled factor; tie goes to lower factor"}
        figure_hashes = {path.name: _sha(path) for path in sorted(figure_dir.glob("*.png"))}
        qa_summary = []
        for run_id, item in sorted(run_meta.items()):
            qa_summary.append({"run_id": run_id, "scenario_key": item["scenario_key"],
                               "eligible": item["eligible"], "reason_codes": item["reason_codes"],
                               "continuity_error_pct": item["hydraulic_quality"]["continuity_error_pct"],
                               "drainage_state": item["hydraulic_quality"]["drainage_state"],
                               "finite_coverage_ok": item["hydraulic_quality"]["finite_coverage_ok"]})
        summary = {
            "schema_version": "chico_norte_novel_hydrograph_results_v1",
            "experiment_root": str(run_root.resolve()),
            "design_sha256": _sha(design_path),
            "protocol_sha256": _sha(run_root / "protocol.json"),
            "source_manifest_sha256": _sha(Path(staged["manifest_path"])),
            "total_scenarios": len(plan),
            "eligible_scenarios": len(score_reports["xgboost"]["eligible_run_ids"]),
            "excluded_scenarios": len(plan) - len(score_reports["xgboost"]["eligible_run_ids"]),
            "models": {family: {"candidate_metadata_sha256": _sha(candidate_dir / "metadata.json"),
                                "registration_sha256": _sha(transfer_dirs[family] / "registration.json"),
                                "prediction_sha256": _sha(transfer_dirs[family] / "predictions.csv"),
                                "verified_scores_sha256": _sha(transfer_dirs[family] / "verified_scores.json"),
                                "overall": score_reports[family]["overall"],
                                "by_event": score_reports[family]["by_event"],
                                "by_scenario": score_reports[family]["by_scenario"]}
                       for family, candidate_dir in MODEL_CANDIDATES.items()},
            "qa_by_scenario": qa_summary,
            "map_scenario_selection": selected_maps,
            "figures_directory": str(figure_dir),
            "figure_sha256": figure_hashes,
            "node_results_rows_after_scoring": int(conn.execute("SELECT COUNT(*) FROM node_results").fetchone()[0]),
            "factor_total_m3_by_shape": totals.to_dict(orient="records"),
        }
        summary_path = run_root / "evaluation_results.json"
        _write_json(summary_path, summary)
        completion = {
            "status": "complete",
            "results_sha256": _sha(summary_path),
            "figure_sha256": figure_hashes,
            "sqlite_node_results_rows": 0,
            "scenario_count": len(plan),
            "eligible_scenario_count": summary["eligible_scenarios"],
            "excluded_scenario_count": summary["excluded_scenarios"],
        }
        _write_json(run_root / "evaluation_completion.json", completion)
        return summary
    finally:
        conn.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args(argv)
    preflight = _preflight()
    if args.dry_run:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_root = EXPERIMENTS / "chico_norte_novel_hydrograph_eval_20261102" / f"run_{timestamp}"
    root = args.run_root.resolve() if args.run_root else default_root.resolve()
    result = run_evaluation(root, args.figure_dir)
    print(json.dumps({
        "experiment_root": result["experiment_root"],
        "scenarios": result["total_scenarios"],
        "eligible": result["eligible_scenarios"],
        "excluded": result["excluded_scenarios"],
        "results": str(root / "evaluation_results.json"),
        "figures": result["figures_directory"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
