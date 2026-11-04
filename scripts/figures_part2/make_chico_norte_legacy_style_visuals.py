"""Render Chico Norte transfer figures using the original full-run visual style.

The data are loaded through the same sealed, hash-verifying pipeline used by
``make_chico_norte_transfer_figures.py``. This renderer writes only aggregate
images and a manifest; node-level labels remain transient in memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

from make_chico_norte_transfer_figures import (
    FLOOD_THRESHOLD_M3,
    MODEL_COLORS,
    MODEL_NAMES,
    MODEL_ORDER,
    OUTPUT_DIR,
    SOURCE_ROOT,
    TRANSFER_ROOTS,
    _class_column,
    _figure_metrics,
    _routed_column,
    _sha256,
    load_verified_transfer,
    parse_conduits,
    parse_coordinates,
    select_representative_scenarios,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_STYLE_DIR = OUTPUT_DIR / "estilo_corrida_completa"


def make_parity_figure(frame, output_path: Path, *, flooded_only: bool = False) -> dict:
    """Draw two model panels in the original scatter style, optionally for wet labels only."""
    plotted = frame.loc[frame["vol_inundacion_m3"].ge(FLOOD_THRESHOLD_M3)].copy() if flooded_only else frame
    actual = plotted["vol_inundacion_m3"].to_numpy(dtype=float)
    maximum = max(float(actual.max()), *(float(plotted[_routed_column(m)].max()) for m in MODEL_ORDER), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True, sharey=True, constrained_layout=True)
    results = {}
    for ax, model in zip(axes, MODEL_ORDER):
        pred = plotted[_routed_column(model)].to_numpy(dtype=float)
        limit = maximum * 1.05
        # The original complete-run figures use node scatter with symlog axes.
        # Small, translucent markers keep the much larger transfer population legible.
        ax.scatter(actual, pred, alpha=0.035, s=2.0, color="steelblue", linewidths=0, rasterized=True)
        ref = np.linspace(0.0, limit, 200)
        ax.plot(ref, ref, "r--", linewidth=0.9, label="y = x · SWMM = predicción")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlim(0.0, limit)
        ax.set_ylim(0.0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(MODEL_NAMES[model], fontweight="bold")
        ax.set_xlabel("SWMM (m³)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
        ax.legend(fontsize=7, loc="lower right")

        metrics = _figure_metrics(plotted, model)
        sst = float(np.square(actual - actual.mean()).sum())
        r2 = float(1.0 - np.square(pred - actual).sum() / sst) if sst else 0.0
        both_zero_pct = float(np.mean((actual == 0) & (pred == 0)) * 100.0)
        results[model] = {**metrics, "r2_descriptive": r2, "both_zero_pct": both_zero_pct}
        ax.text(
            0.03,
            0.87,
            f"R² = {r2:.3f}  MAE = {metrics['mae_m3']:.2f} m³\nRMSE = {metrics['rmse_m3']:.2f} m³\nn = {metrics['n']:,} pares",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
    axes[0].set_ylabel("Volumen predicho enrutado (m³)")
    fig.suptitle(
        f"Paridad nodal · Chico Norte · {len(plotted):,} pares" + (" inundados por SWMM" if flooded_only else " de 787 corridas elegibles"),
        fontsize=14,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return results


def make_flood_map(frame, case: dict, run_id: int, inp_path: Path, output_path: Path) -> dict:
    """Draw SWMM, XGBoost and MLP panels in the established network-map style."""
    rows = frame.loc[frame["run_id"].eq(int(run_id))].copy()
    if len(rows) != 522 or rows["node_id"].duplicated().any():
        raise ValueError(f"Selected scenario {case['scenario_key']} lacks 522 unique nodes")
    coordinates = parse_coordinates(inp_path)
    conduits = parse_conduits(inp_path)
    if not set(rows["node_id"]).issubset(coordinates):
        raise ValueError("The Chico Norte INP coordinates do not cover every node")

    maximum = max(
        1.0,
        float(rows["vol_inundacion_m3"].max()),
        *(float(rows[_routed_column(model)].max()) for model in MODEL_ORDER),
    )
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=maximum)
    cmap = plt.get_cmap("YlOrRd")
    positions = rows["node_id"].map(coordinates)
    xs = np.array([point[0] for point in positions])
    ys = np.array([point[1] for point in positions])
    n_swmm_flooded = int((rows["vol_inundacion_m3"] >= FLOOD_THRESHOLD_M3).sum())
    columns = [("vol_inundacion_m3", "SWMM (referencia)", "#455a64", None)]
    columns.extend((_routed_column(model), MODEL_NAMES[model], MODEL_COLORS[model], model) for model in MODEL_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.7))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.79, bottom=0.25, wspace=0.035)
    mappable = None
    summaries = {}
    for ax, (column, title, edge_color, model) in zip(axes, columns):
        for _link_id, start, end in conduits:
            if start in coordinates and end in coordinates:
                ax.plot(
                    [coordinates[start][0], coordinates[end][0]],
                    [coordinates[start][1], coordinates[end][1]],
                    color="#b0bec5",
                    lw=0.7,
                    zorder=1,
                )
        values = rows[column].to_numpy(dtype=float)
        if model is None:
            wet = values >= FLOOD_THRESHOLD_M3
        else:
            wet = rows[_class_column(model)].to_numpy(dtype=int).astype(bool)
        dry = ~wet
        ax.scatter(xs[dry], ys[dry], s=10, color="#cfd8dc", edgecolors="#90a4ae", linewidths=0.4, zorder=2)
        mappable = ax.scatter(
            xs[wet],
            ys[wet],
            s=30 + 220 * np.clip(values[wet] / maximum, 0, 1) ** 0.5,
            c=values[wet],
            cmap=cmap,
            norm=norm,
            edgecolors=edge_color,
            linewidths=0.9,
            zorder=3,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        count = int(wet.sum())
        total = float(values.sum())
        if model is None:
            summary = f"Total SWMM: {total:,.0f} m³\n{count} nodos inundados"
        else:
            metrics = _figure_metrics(rows, model)
            actual_class = rows["vol_inundacion_m3"].to_numpy(dtype=float) >= FLOOD_THRESHOLD_M3
            pred_class = rows[_class_column(model)].to_numpy(dtype=int).astype(bool)
            fp = int((~actual_class & pred_class).sum())
            fn = int((actual_class & ~pred_class).sum())
            summary = (
                f"Total predicho: {total:,.0f} m³ ({(total / max(float(rows['vol_inundacion_m3'].sum()), 1e-12) - 1) * 100:+.1f}%)\n"
                f"{count} nodos inundados · {fp} falsos positivos · {fn} omitidos\n"
                f"RMSE nodal: {metrics['rmse_m3']:.1f} m³"
            )
            summaries[model] = {**metrics, "false_positives": fp, "false_negatives": fn}
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.text(0.5, -0.03, summary, transform=ax.transAxes, ha="center", va="top", fontsize=8)

    cax = fig.add_axes([0.31, 0.105, 0.38, 0.035])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    ticks = [t for t in (0, 100, 1_000, 10_000, 100_000) if t <= maximum]
    cbar.set_ticks(ticks)
    cbar.set_label("Volumen de inundación por nodo (m³; escala de raíz, común a los tres paneles)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"Mapas de inundación · Chico Norte · escenario {case['label']} · p{int(case['quantile'] * 100)}\n"
        f"Volumen SWMM total: {case['total_swmm_m3']:,.0f} m³ · {n_swmm_flooded} nodos inundados",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return summaries


def main() -> None:
    if LEGACY_STYLE_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite presentation-style figures: {LEGACY_STYLE_DIR}")
    frame, totals, provenance = load_verified_transfer(TRANSFER_ROOTS, SOURCE_ROOT)
    LEGACY_STYLE_DIR.mkdir(parents=True)
    parity_path = LEGACY_STYLE_DIR / "paridad_transferencia_chico_norte.png"
    parity_wet_path = LEGACY_STYLE_DIR / "paridad_nodos_inundados_chico_norte.png"
    parity = make_parity_figure(frame, parity_path)
    parity_wet = make_parity_figure(frame, parity_wet_path, flooded_only=True)
    cases = select_representative_scenarios(totals)
    registration = json.loads((Path(__file__).resolve().parents[2] / "outputs" / "bench" / "multinetwork" / "20261030T005341Z-813323e3ef75" / "A_to_B_frozen" / "registration.json").read_text(encoding="utf-8"))
    inp_path = Path(registration["b_inp_path"]).resolve(strict=True)
    scenario_manifest = []
    for case in cases:
        run_id = int(totals.set_index("scenario_key").loc[case["scenario_key"], "run_id"])
        image_path = LEGACY_STYLE_DIR / f"mapa_{case['label']}_chico_norte.png"
        model_stats = make_flood_map(frame, case, run_id, inp_path, image_path)
        scenario_manifest.append({**case, "run_id": run_id, "models": model_stats})
    artifacts = [parity_path, parity_wet_path, *(LEGACY_STYLE_DIR / f"mapa_{case['label']}_chico_norte.png" for case in cases)]
    manifest = {
        "schema_version": "chico_norte_transfer_legacy_style_v1",
        "style_reference": "scripts/figures_part2/make_parity_and_maps.py (full-run visuals)",
        "source_transfer_figure_manifest_sha256": _sha256(OUTPUT_DIR / "figure_manifest.json"),
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "frozen_batch_manifest_sha256": provenance["frozen_batch_manifest_sha256"],
        "network_sha256": provenance["network_sha256"],
        "labels_sha256": provenance["labels_sha256"],
        "eligible_runs": provenance["eligible_runs"],
        "excluded_runs": provenance["excluded_runs"],
        "run_node_rows": provenance["rows"],
        "label_persistence": "transient_from_hash_verified_rpt; no_node_level_target_file_written",
        "parity": {
            "style": "two-panel raw scatter, symlog axes, steelblue markers, dashed y=x line",
            "all_pairs": {"figure": parity_path.name, "models": parity},
            "swmm_flooded_subset": {"figure": parity_wet_path.name, "n": int(frame['vol_inundacion_m3'].ge(FLOOD_THRESHOLD_M3).sum()), "models": parity_wet},
        },
        "scenario_selection": scenario_manifest,
        "artifacts": [{"path": path.name, "sha256": _sha256(path), "byte_size": path.stat().st_size} for path in artifacts],
    }
    (LEGACY_STYLE_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Generated original-style figures: {LEGACY_STYLE_DIR}")
    print(f"Verified rows: {provenance['rows']}; eligible: {provenance['eligible_runs']}; labels SHA-256: {provenance['labels_sha256']}")


if __name__ == "__main__":
    main()
