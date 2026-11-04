"""Build separate, full-run-style figures for the frozen Chico Norte evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RUN = REPO / "outputs/experiments/chico_norte_novel_hydrograph_eval_20261102/run_20261103T034108Z"
OUTPUT = REPO / "docs/figures_chico_norte_novel_eval_2026_11_02_presentacion"
INP = REPO / "data/networks/chico_norte_93_qx1/SWMM - Chico Norte 93% - Qx1.00.inp"
SHAPE_DIR = REPO / "data/hydrograph_validation/chico_norte_nuevos_2026_11_02"
BASE_FIGURES = REPO / "docs/figures_chico_norte_novel_eval_2026_11_02"
MAP_FACTORS = (0.7, 2.7, 4.3)
FLOOD_THRESHOLD_M3 = 1.0
MODEL_COLORS = {"xgboost": "#4386C6", "mlp": "#E87525"}
MODEL_LABELS = {"xgboost": "XGBoost", "mlp": "MLP (5 semillas)"}
MODEL_VOLUME_COLUMNS = {"xgboost": "xgb_volume_m3", "mlp": "mlp_volume_m3"}
MODEL_CLASS_COLUMNS = {"xgboost": "xgb_class", "mlp": "mlp_class"}
MAP_MODEL_ORDER = ("swmm", "mlp", "xgboost")
SHAPE_LABELS = {
    "novel_20261102_cascade_peaks": "Pulsos dobles separados",
    "novel_20261102_sustained_late_surge": "Meseta sostenida y pico tardío",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_map_scenarios(available, factors=MAP_FACTORS):
    """Select the requested low, middle and high factors for every shape."""
    rows = list(available)
    shapes = list(dict.fromkeys(str(row["shape_id"]) for row in rows))
    selected = []
    for shape in shapes:
        for factor in factors:
            matches = [
                row for row in rows
                if str(row["shape_id"]) == shape
                and abs(float(row["factor_mult"]) - float(factor)) < 1e-9
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"missing requested map scenario or duplicate: {shape} factor {factor}"
                )
            selected.append(dict(matches[0]))
    if not selected or len(shapes) != 2:
        raise ValueError("expected exactly two hydrograph shapes for map selection")
    return selected


def expand_map_scenarios(available, factors=MAP_FACTORS):
    """Return one independent SWMM, MLP and XGBoost map for each case."""
    expanded = []
    for scenario in select_map_scenarios(available, factors=factors):
        for model_key in MAP_MODEL_ORDER:
            model_name = {"swmm": "SWMM", "mlp": "MLP", "xgboost": "XGBoost"}[model_key]
            expanded.append({**scenario, "model_key": model_key, "model_name": model_name})
    return expanded


def _prediction_paths() -> dict[str, Path]:
    paths = {}
    for family in ("xgboost", "mlp"):
        candidates = list((RUN / "predictions" / family).glob("*/A_to_B_frozen/predictions.csv"))
        if len(candidates) != 1:
            raise ValueError(f"expected one sealed {family} predictions file, got {len(candidates)}")
        path = candidates[0]
        seal = json.loads((path.parent / "prediction_seal.json").read_text(encoding="utf-8"))
        if _sha(path) != seal["predictions_sha256"]:
            raise ValueError(f"{family} predictions do not match their seal")
        paths[family] = path
    return paths


def load_verified_frame() -> tuple[pd.DataFrame, dict, dict]:
    """Rebuild evaluation rows from sealed predictions and hash-verified SWMM RPTs."""
    completion = json.loads((RUN / "evaluation_completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("eligible_scenario_count") != 10:
        raise ValueError("the frozen ten-scenario evaluation is not complete")
    for filename, expected in completion["figure_sha256"].items():
        if _sha(BASE_FIGURES / filename) != expected:
            raise ValueError(f"original evidence figure changed: {filename}")

    predictions = {
        family: pd.read_csv(path, dtype={"node_id": str}, float_precision="round_trip")
        for family, path in _prediction_paths().items()
    }
    keys = ["run_id", "node_id"]
    if not predictions["xgboost"][keys].equals(predictions["mlp"][keys]):
        raise ValueError("the two frozen candidates do not share identical row keys")

    manifest = json.loads((RUN / "experiment_source_manifest.json").read_text(encoding="utf-8"))
    from swmm_resilience.extraction.labels import extract_labels
    from swmm_resilience.extraction.static_features import extract_static_features

    node_ids = extract_static_features(INP)["node_id"].astype(str).tolist()
    truth_parts = []
    metadata = {}
    for entry in manifest["run_order"]:
        if not entry["eligible"]:
            continue
        run_id = int(entry["run_id"])
        rpt_info = entry["artifacts"]["swmm_rpt"]
        rpt_path = Path(rpt_info["path"])
        if not rpt_path.is_absolute():
            rpt_path = RUN / rpt_path
        if _sha(rpt_path) != rpt_info["sha256"]:
            raise ValueError(f"RPT hash mismatch for run {run_id}")
        labels = extract_labels(rpt_path, node_ids, FLOOD_THRESHOLD_M3)
        labels["run_id"] = run_id
        truth_parts.append(labels[["run_id", "node_id", "inunda", "vol_inundacion_m3"]])
        metadata[run_id] = {
            "scenario_key": entry["scenario_key"],
            "shape_id": entry["scenario"]["shape_id"],
            "factor_mult": float(entry["scenario"]["factor_mult"]),
        }
    truth = pd.concat(truth_parts, ignore_index=True)
    xgb, mlp = predictions["xgboost"], predictions["mlp"]
    joined = truth.merge(xgb, on=keys, how="inner", validate="one_to_one", suffixes=("", "_xgb"))
    mlp_cols = keys + ["y_prob_clf", "y_pred_clf", "volume_routed_m3"]
    joined = joined.merge(mlp[mlp_cols], on=keys, how="inner", validate="one_to_one", suffixes=("_xgb", "_mlp"))
    joined = joined.rename(columns={
        "y_prob_clf_mlp": "mlp_probability", "y_pred_clf_mlp": "mlp_class",
        "volume_routed_m3_mlp": "mlp_volume_m3",
        "y_prob_clf_xgb": "xgb_probability", "y_pred_clf_xgb": "xgb_class",
        "volume_routed_m3_xgb": "xgb_volume_m3",
    })
    joined["scenario_key"] = joined["run_id"].map(lambda run_id: metadata[int(run_id)]["scenario_key"])
    joined["shape_id"] = joined["run_id"].map(lambda run_id: metadata[int(run_id)]["shape_id"])
    joined["factor_mult"] = joined["run_id"].map(lambda run_id: metadata[int(run_id)]["factor_mult"])
    if len(joined) != 5220 or joined.duplicated(keys).any():
        raise ValueError(f"expected 5,220 unique eligible pairs, got {len(joined)}")
    return joined, metadata, manifest


def plot_hydrograph_series(ax, times: np.ndarray, values: np.ndarray) -> None:
    """Draw a line hydrograph matching the established full-run figure style."""
    color = "#385F8B"
    ax.plot(times, values, color=color, linewidth=2.5, marker="o", markersize=5)
    ax.fill_between(times, values, alpha=0.15, color=color)
    ax.set_xlabel("Tiempo desde el inicio del evento (h)")
    ax.set_ylabel("Caudal normalizado (q/qmáx)")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, float(times[-1]))
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Hidrograma sintético de entrada", fontweight="bold", color="#172F4D")


def draw_hydrograph(path: Path, shape_id: str) -> None:
    data = pd.read_csv(path)
    times = data["time_h"].to_numpy(dtype=float)
    values = data["q_norm"].to_numpy(dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError(f"invalid normalized hydrograph: {path}")
    if np.any(np.diff(times) <= 0) or np.any(values < 0):
        raise ValueError(f"hydrograph times must increase and flows must be non-negative: {path}")
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    plot_hydrograph_series(ax, times, values)
    ax.set_title(f"Hidrograma sintético de entrada · {SHAPE_LABELS[shape_id]}",
                 fontweight="bold", color="#172F4D")
    fig.savefig(OUTPUT / f"hidrograma_{shape_id}.png", dpi=220, facecolor="white")
    plt.close(fig)


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residual = predicted - actual
    sst = float(np.square(actual - actual.mean()).sum())
    return {
        "rmse_m3": float(np.sqrt(np.square(residual).mean())),
        "mae_m3": float(np.abs(residual).mean()),
        "r2": float(1 - np.square(residual).sum() / sst) if sst else 0.0,
        "n": int(len(actual)),
    }


def draw_parity(frame: pd.DataFrame, family: str) -> dict:
    actual = frame["vol_inundacion_m3"].to_numpy(dtype=float)
    predicted = frame[MODEL_VOLUME_COLUMNS[family]].to_numpy(dtype=float)
    metrics = _metric_values(actual, predicted)
    maximum = max(float(actual.max()), float(predicted.max()), 1.0) * 1.05
    upper = float(np.log10(1.0 + maximum))
    x = np.log10(1.0 + actual)
    y = np.log10(1.0 + predicted)
    fig, ax = plt.subplots(figsize=(7.5, 6.3), constrained_layout=True)
    ax.scatter(x, y, alpha=0.33, s=16, color=MODEL_COLORS[family], linewidths=0,
               rasterized=True, label=MODEL_LABELS[family])
    ax.plot([0, upper], [0, upper], "--", color="#59636E", linewidth=1.1, label="y = x")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    tick_values = [0, 1, 10, 100, 1_000, 10_000, 100_000]
    tick_values = [value for value in tick_values if np.log10(1 + value) <= upper + 1e-9]
    ax.set_xticks([np.log10(1 + value) for value in tick_values],
                  [f"{value:,.0f}" for value in tick_values])
    ax.set_yticks([np.log10(1 + value) for value in tick_values],
                  [f"{value:,.0f}" for value in tick_values])
    ax.set_xlabel("Volumen SWMM (m³)")
    ax.set_ylabel("Volumen predicho (m³)")
    ax.set_title(f"Paridad global · {MODEL_LABELS[family]}", loc="left", fontweight="bold")
    ax.grid(True, color="#CFD8E3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.04, 0.96,
            f"RMSE = {metrics['rmse_m3']:.2f} m³\nMAE = {metrics['mae_m3']:.2f} m³\n"
            f"R² = {metrics['r2']:.3f}\nn = {metrics['n']:,} pares",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB", "alpha": 0.94})
    fig.savefig(OUTPUT / f"paridad_global_{family}.png", dpi=220, facecolor="white")
    plt.close(fig)
    return metrics


def plot_flood_map(ax, coordinates, conduits, node_ids, values, wet, model_name):
    """Draw only inundated nodes over the network, without dry-node or error symbols."""
    values = np.asarray(values, dtype=float)
    wet = np.asarray(wet, dtype=bool)
    node_ids = list(node_ids)
    vmax = max(float(values[wet].max()) if wet.any() else 0.0, 1.0)
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd")
    for _, start, end in conduits:
        if start in coordinates and end in coordinates:
            ax.plot([coordinates[start][0], coordinates[end][0]],
                    [coordinates[start][1], coordinates[end][1]],
                    color="#B0BEC5", lw=0.7, zorder=1)
    xs = np.array([coordinates[node][0] for node in node_ids])
    ys = np.array([coordinates[node][1] for node in node_ids])
    mappable = ax.scatter(xs[wet], ys[wet], c=values[wet], cmap=cmap, norm=norm,
                          s=30 + 220 * np.sqrt(np.clip(values[wet] / vmax, 0, 1)),
                          edgecolors="none", linewidths=0, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    ax.set_title(model_name, fontweight="bold", fontsize=14)
    return mappable


def draw_network_map(frame: pd.DataFrame, row: dict, inp_path: Path, out_path: Path) -> dict:
    # Reuse the INP parsers from the established Chico Norte map pipeline.
    figure_dir = str(REPO / "scripts/figures_part2")
    if figure_dir not in sys.path:
        sys.path.insert(0, figure_dir)
    from make_chico_norte_transfer_figures import parse_conduits, parse_coordinates

    rows = frame.loc[frame["scenario_key"] == row["scenario_key"]].copy()
    if len(rows) != 522 or rows["node_id"].duplicated().any():
        raise ValueError(f"map scenario {row['scenario_key']} does not have 522 unique nodes")
    coordinates, conduits = parse_coordinates(inp_path), parse_conduits(inp_path)
    if not set(rows["node_id"]).issubset(coordinates):
        raise ValueError("network coordinates do not include every scored node")

    model = row["model_key"]
    if model == "swmm":
        values = rows["vol_inundacion_m3"].to_numpy(dtype=float)
        wet = values >= FLOOD_THRESHOLD_M3
    else:
        values = rows[MODEL_VOLUME_COLUMNS[model]].to_numpy(dtype=float)
        wet = rows[MODEL_CLASS_COLUMNS[model]].to_numpy(dtype=int).astype(bool)
    fig = plt.figure(figsize=(13.0, 6.2))
    ax = fig.add_axes([0.025, 0.12, 0.82, 0.78])
    mappable = plot_flood_map(ax, coordinates, conduits, rows["node_id"], values,
                              wet, row["model_name"])
    cax = fig.add_axes([0.89, 0.18, 0.02, 0.62])
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_label("Volumen de inundación por nodo (m³)", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    total, count = float(values.sum()), int(wet.sum())
    fig.suptitle(f"{SHAPE_LABELS[row['shape_id']]} · factor {float(row['factor_mult']):.1f}",
                 fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.43, 0.055, f"Volumen total: {total:,.0f} m³ · nodos inundados: {count}",
             ha="center", fontsize=10)
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    return {
        "scenario_key": row["scenario_key"], "shape_id": row["shape_id"],
        "factor_mult": float(row["factor_mult"]), "model_key": model,
        "model_name": row["model_name"], "filename": out_path.name,
        "total_flood_volume_m3": total, "flooded_nodes": count,
        "color_scale": {"colormap": mappable.get_cmap().name, "vmin_m3": 0.0,
                        "vmax_m3": float(mappable.norm.vmax), "normalization": "square_root"},
    }


def build_figures() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing presentation figures: {OUTPUT}")
    frame, metadata, manifest = load_verified_frame()
    OUTPUT.mkdir(parents=True)
    for shape_id in SHAPE_LABELS:
        draw_hydrograph(SHAPE_DIR / f"{shape_id}.csv", shape_id)
    parity = {family: draw_parity(frame, family) for family in ("xgboost", "mlp")}
    available = [
        {"shape_id": item["shape_id"], "factor_mult": item["factor_mult"],
         "scenario_key": item["scenario_key"]}
        for item in metadata.values()
    ]
    selected = expand_map_scenarios(available)
    inp_record = json.loads((RUN / "evaluation_design.json").read_text(encoding="utf-8"))["network_inp"]
    inp_path = Path(inp_record["path"])
    if _sha(inp_path) != inp_record["sha256"]:
        raise ValueError("Chico Norte INP hash differs from the frozen design")
    maps = []
    for row in selected:
        filename = (f"mapa_{row['shape_id']}_f{float(row['factor_mult']):.1f}_"
                    f"{row['model_key']}.png")
        maps.append(draw_network_map(frame, row, inp_path, OUTPUT / filename))

    # The user approved keeping the existing model-vs-SWMM factor lines unchanged.
    volume_figure = BASE_FIGURES / "volumen_total_por_factor.png"
    completion = json.loads((RUN / "evaluation_completion.json").read_text(encoding="utf-8"))
    if _sha(volume_figure) != completion["figure_sha256"][volume_figure.name]:
        raise ValueError("the approved factor-comparison figure changed")
    (OUTPUT / volume_figure.name).write_bytes(volume_figure.read_bytes())

    files = sorted(path for path in OUTPUT.glob("*.png"))
    result = {
        "schema_version": "chico_norte_novel_presentation_figures_v2",
        "source_evaluation_results_sha256": _sha(RUN / "evaluation_results.json"),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "network_inp_sha256": inp_record["sha256"],
        "parity_metrics": parity,
        "map_factors": list(MAP_FACTORS),
        "map_style": "one standalone network map per model and scenario; only inundated nodes are drawn; independent YlOrRd flood-volume scale; no dry-node or error markers",
        "maps": maps,
        "factor_lines_source": str(volume_figure.relative_to(REPO)).replace("\\", "/"),
        "figure_sha256": {path.name: _sha(path) for path in files},
    }
    (OUTPUT / "presentation_figures_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (OUTPUT / "INDEX.md").write_text(
        "# Figuras para la presentación · Chico Norte\n\n"
        "Material basado en la evaluación congelada y en predicciones/RPT verificados por hash. "
        "No se ejecutó SWMM ni se reentrenaron modelos para construir estas figuras.\n\n"
        "La presentación no conserva las tres diapositivas comparativas antiguas del deck base, "
        "porque sus mapas incluían marcas de falsos positivos y omisiones que no corresponden al formato solicitado.\n\n"
        "## Orden y estilo\n\n"
        "1. Dos hidrogramas separados, como curvas azules con marcadores y relleno claro, siguiendo el estilo del flujo completo.\n"
        "2. Dieciocho mapas independientes: SWMM, MLP y XGBoost para cada uno de seis escenarios "
        "(factores 0,7, 2,7 y 4,3 en ambas formas). Cada mapa dibuja la red en gris y solo los nodos inundados, "
        "los volúmenes inundados en `YlOrRd` y su propia barra de color. No se superponen marcas de falsas alarmas ni omisiones. "
        "Las escalas son independientes y los colores no se comparan entre mapas.\n"
        "3. Dos paridades globales de puntos, separadas por modelo, con los colores de XGBoost y MLP, línea de igualdad "
        "y ejes `log10(1 + volumen)`.\n"
        "4. La gráfica de rectas de volumen total por factor, copiada sin cambios desde la evaluación original.\n\n"
        "El manifiesto `presentation_figures_manifest.json` registra hashes de imágenes, escenarios y resumen por mapa. "
        "El guion y los resultados están en [el reporte de evaluación](../../superpowers/reports/2026-11-02-evaluacion-hidrogramas-nuevos-chico-norte.md) "
        "y [el guion oral](../../resultados/guion_presentacion_2026-11-02-chico-norte.md).\n",
        encoding="utf-8")
    return result


if __name__ == "__main__":
    summary = build_figures()
    print(f"Generated {len(summary['figure_sha256'])} presentation figures in {OUTPUT}")
