"""Parity plots, per-shape errors and flood maps for the tuned Part 2 models.

Models (tuned, LeaveOneShapeOut out-of-fold predictions, one fold per shape):
XGBoost, MLP and approximate SVM. Every figure is built from the persisted
OOF files of the Part 2 run; nothing is retrained and nothing outside
``docs/figures_part2/`` is written.

Predicted routed volume of a node-scenario row = ``y_pred_reg`` where
``y_pred_clf == 1`` and 0 otherwise (the end-to-end prediction that the
evaluation scores). SWMM volume = ``vol_inundacion_m3`` of the PreparedDataset.

Usage (from the repository root)::

    ../../../venv/Scripts/python.exe scripts/figures_part2/make_parity_and_maps.py

The multi-panel maps and the model-aware labels are drawn here with matplotlib
(the shared plotting modules hardcode "XGBoost" in their axis labels); the
geometry comes from the project's ``.inp`` parser.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from swmm_resilience.ml.bench.preprocess import resolve_prepared  # noqa: E402
from swmm_resilience.visualization._inp_parser import (  # noqa: E402
    parse_conduits,
    parse_coordinates,
)

PREP_ID = "ca706e27654e0cf5"
PREPARED_DIR = REPO_ROOT / "outputs" / "bench" / "prepared_v4_21"
OOF_ROOT = REPO_ROOT / "outputs" / "bench" / "parte2" / PREP_ID
INP_PATH = (
    REPO_ROOT / "data" / "networks" / "chico_hydro-qx1"
    / "SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
)
CSV_PROTOCOLS = (
    REPO_ROOT / "docs" / "auditoria" / "correcciones" / "datos_parte2"
    / "comparacion_protocolos.csv"
)
OUT_DIR = REPO_ROOT / "docs" / "figures_part2"

TARGET_FACTOR = 3.0
MIN_FLOODED_ROWS = 20
HIGHLIGHT_SHAPES = ("shape_quasi_uniform_block", "common_bogota")
DPI = 200

# (family key, label on the figures, colour, CSV model name)
MODELS = (
    ("xgboost", "XGBoost", "#1f6fb4", "xgboost_tuned"),
    ("mlp", "MLP", "#e8830c", "mlp_tuned"),
    ("svm", "Approx. SVM", "#2a9d4a", "svm_tuned"),
)
LABEL = {k: lab for k, lab, _, _ in MODELS}
SWMM_COLOR = "#4d4d4d"
SUFFIX = "tuned, LeaveOneShapeOut"

FS_TITLE, FS_LABEL, FS_TICK, FS_NOTE = 12, 10, 9, 9
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_NOTE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    }
)


# ----------------------------------------------------------------------------- data


def load_frame() -> pd.DataFrame:
    """One row per node-scenario with the SWMM volume and the 3 predictions."""
    prepared = resolve_prepared(PREPARED_DIR, PREP_ID)
    df = prepared.keys[["node_id", "shape_id", "factor_mult"]].copy()
    df["vol_swmm"] = prepared.y_reg.to_numpy(dtype=float)
    df["inunda"] = prepared.y_clf.to_numpy(dtype=int)
    for fam, _, _, _ in MODELS:
        oof = pd.read_parquet(OOF_ROOT / fam / "tuned" / "oof.parquet")
        oof = oof[oof["protocol"] == "LeaveOneShapeOut"].sort_values("sample_idx")
        if len(oof) != len(df) or not (oof["sample_idx"].to_numpy() == np.arange(len(df))).all():
            raise ValueError(f"OOF of {fam} does not cover the dataset exactly once.")
        routed = np.where(oof["y_pred_clf"].to_numpy() == 1, oof["y_pred_reg"].to_numpy(), 0.0)
        df[f"pred_{fam}"] = routed
    return df


def pooled_metrics(df: pd.DataFrame, fam: str) -> dict:
    y = df["vol_swmm"].to_numpy()
    p = df[f"pred_{fam}"].to_numpy()
    sse = float(((p - y) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "rmse_m3": float(np.sqrt(sse / len(y))),
        "r2": 1.0 - sse / sst,
        "mae_m3": float(np.abs(p - y).mean()),
        "n_rows": int(len(y)),
        "total_error_pct": float((p.sum() / y.sum() - 1.0) * 100.0),
    }


def check_against_csv(metrics: dict) -> None:
    """The pooled RMSE must equal the audited CSV (within rounding)."""
    ref = pd.read_csv(CSV_PROTOCOLS)
    ref = ref[ref["protocolo"] == "LeaveOneShapeOut"].set_index("modelo")
    for fam, _, _, csv_name in MODELS:
        csv_rmse = float(ref.loc[csv_name, "rmse_vol_agregado_m3"])
        ours = metrics[fam]["rmse_m3"]
        print(f"  check {csv_name}: figure RMSE {ours:.4f} vs CSV {csv_rmse:.4f}")
        if abs(ours - csv_rmse) > 0.01:
            raise AssertionError(f"RMSE mismatch for {csv_name}: {ours} vs {csv_rmse}")
        metrics[fam]["rmse_csv_m3"] = csv_rmse


def scenario_totals(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["vol_swmm"] + [f"pred_{f}" for f, _, _, _ in MODELS]
    return df.groupby(["shape_id", "factor_mult"], as_index=False)[cols].sum()


def shape_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shape, g in df.groupby("shape_id"):
        row = {"shape_id": shape, "vol_swmm": g["vol_swmm"].sum(),
               "n_flooded_rows": int(g["inunda"].sum())}
        for fam, _, _, _ in MODELS:
            p = g[f"pred_{fam}"].to_numpy()
            row[f"sse_{fam}"] = float(((p - g["vol_swmm"].to_numpy()) ** 2).sum())
            row[f"err_pct_{fam}"] = float((p.sum() / g["vol_swmm"].sum() - 1.0) * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_factor(df: pd.DataFrame, shape: str) -> float:
    """Factor of the shape closest to TARGET_FACTOR (ties go to the lower one)."""
    avail = np.sort(df.loc[df["shape_id"] == shape, "factor_mult"].unique())
    return float(avail[np.argmin(np.abs(avail - TARGET_FACTOR))])


def select_scenarios(df: pd.DataFrame, shapes: pd.DataFrame) -> list[dict]:
    """Documented, non-cherry-picked selection (all rules use tuned XGBoost).

    Error = total squared error (sum over every row of the shape, all factors)
    of the tuned XGBoost out-of-fold prediction.
      max_error    : shape with the largest error.
      median_error : shape at position n//2 of the ascending ranking (n = 40, the
                     upper median).
      min_error    : lowest error among shapes whose displayed scenario (factor
                     nearest to 3.0) has at least 20 SWMM-flooded rows.
    """
    ranked = shapes.sort_values("sse_xgboost", ascending=True).reset_index(drop=True)
    picks = {
        "max_error": ranked.iloc[-1]["shape_id"],
        "median_error": ranked.iloc[len(ranked) // 2]["shape_id"],
    }
    for _, r in ranked.iterrows():
        f = resolve_factor(df, r["shape_id"])
        sub = df[(df["shape_id"] == r["shape_id"]) & (df["factor_mult"] == f)]
        if int(sub["inunda"].sum()) >= MIN_FLOODED_ROWS:
            picks["min_error"] = r["shape_id"]
            break
    out = []
    for rule in ("max_error", "median_error", "min_error"):
        shape = picks[rule]
        sse = float(ranked.loc[ranked["shape_id"] == shape, "sse_xgboost"].iloc[0])
        rank = int(ranked.index[ranked["shape_id"] == shape][0]) + 1
        out.append({"rule": rule, "shape": shape, "factor": resolve_factor(df, shape),
                    "sse_xgboost": sse, "rank_ascending_of_40": rank})
    return out


# --------------------------------------------------------------------- figure: parity


def _tick_u(values):
    return np.log10(1.0 + np.asarray(values, dtype=float))


def _white_to(color: str) -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list("w2c", ["#ffffff", color])


def fig_parity_nodes(df: pd.DataFrame, metrics: dict) -> Path:
    """Density of predicted vs SWMM routed volume (log10(1 + v) axes)."""
    y = df["vol_swmm"].to_numpy()
    vmax = max(y.max(), *(df[f"pred_{f}"].max() for f, _, _, _ in MODELS))
    umax = float(_tick_u(vmax)) * 1.03
    edges = np.linspace(0.0, umax, 121)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.9), sharex=True, sharey=True)
    for ax, (fam, lab, col, _) in zip(axes, MODELS):
        p = df[f"pred_{fam}"].to_numpy()
        h, xe, ye = np.histogram2d(_tick_u(y), _tick_u(p), bins=[edges, edges])
        hm = np.ma.masked_equal(h, 0)
        mesh = ax.pcolormesh(xe, ye, hm.T, cmap=_white_to(col),
                             norm=mcolors.LogNorm(vmin=1, vmax=h.max()), rasterized=True)
        ax.plot([0, umax], [0, umax], color=SWMM_COLOR, lw=1.2, ls="--", label="y = x")
        ticks = [t for t in (0, 1, 10, 100, 1000, 10000) if _tick_u(t) <= umax]
        ax.set_xticks(_tick_u(ticks))
        ax.set_xticklabels([f"{t:,}" for t in ticks])
        ax.set_yticks(_tick_u(ticks))
        ax.set_yticklabels([f"{t:,}" for t in ticks])
        ax.set_xlim(0, umax)
        ax.set_ylim(0, umax)
        ax.set_aspect("equal")
        ax.set_xlabel("SWMM flood volume per node-scenario (m³, log1p scale)")
        ax.set_title(f"{lab} ({SUFFIX})", fontweight="bold")
        ax.grid(alpha=0.25)
        m = metrics[fam]
        zz = float(((y == 0) & (p == 0)).mean() * 100)
        m["both_zero_pct"] = zz
        ax.text(
            0.03, 0.97,
            f"RMSE = {m['rmse_m3']:.2f} m³\nR² = {m['r2']:.3f}\nn = {m['n_rows']:,} rows\n"
            f"both zero: {zz:.1f}% of rows",
            transform=ax.transAxes, va="top", fontsize=FS_NOTE,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb", alpha=0.92),
        )
        cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Rows per cell (log scale)", fontsize=FS_NOTE)
        cb.ax.tick_params(labelsize=FS_TICK)
    axes[0].set_ylabel("Predicted routed volume (m³, log1p scale)")
    fig.suptitle(
        "Predicted vs SWMM routed volume per node-scenario row "
        "(154,080 rows; each shape predicted by a model trained without it)",
        fontsize=FS_TITLE + 1, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT_DIR / "parity_nodes_3models.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_parity_totals(tot: pd.DataFrame, numbers: dict) -> Path:
    """Total flooded volume per scenario (shape x factor), log-log."""
    floor = 1.0
    x = tot["vol_swmm"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    hi = max(x.max(), *(tot[f"pred_{f}"].max() for f, _, _, _ in MODELS)) * 1.4
    ax.plot([floor * 0.8, hi], [floor * 0.8, hi], color=SWMM_COLOR, lw=1.2, ls="--",
            label="y = x (perfect agreement)", zorder=1)
    markers = {"xgboost": "o", "mlp": "s", "svm": "^"}
    sizes = {"xgboost": 16, "mlp": 16, "svm": 18}
    valid = x >= floor
    numbers["scenarios_total"] = int(len(tot))
    numbers["scenarios_swmm_ge_1m3"] = int(valid.sum())
    for fam, lab, col, _ in reversed(MODELS):  # XGBoost drawn last (on top)
        p = tot[f"pred_{fam}"].to_numpy()
        ape = np.abs(p[valid] - x[valid]) / x[valid] * 100.0
        med = float(np.median(ape))
        numbers.setdefault("scenario_median_ape_pct", {})[fam] = med
        numbers.setdefault("scenario_p90_ape_pct", {})[fam] = float(np.percentile(ape, 90))
        ax.scatter(np.maximum(x, floor), np.maximum(p, floor), s=sizes[fam], marker=markers[fam],
                   color=col, alpha=0.55, linewidths=0, zorder=3 if fam == "xgboost" else 2,
                   label=f"{lab}: median APE {med:.1f}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(floor * 0.8, hi)
    ax.set_ylim(floor * 0.8, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("SWMM total flood volume per scenario (m³, log scale)")
    ax.set_ylabel("Predicted total flood volume per scenario (m³, log scale)")
    ax.set_title(f"Total flood volume per scenario: predicted vs SWMM\n({SUFFIX})",
                 fontsize=FS_TITLE - 0.5, fontweight="bold")
    ax.grid(alpha=0.25, which="both")
    leg = ax.legend(loc="upper left", framealpha=0.95)
    leg.set_title("Median absolute percentage error (APE)", prop={"size": FS_NOTE})
    ax.text(
        0.98, 0.03,
        f"{len(tot)} scenarios (shape x factor); APE over the {int(valid.sum())} "
        "with SWMM total >= 1 m³;\nvalues below 1 m³ are drawn at 1 m³",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.92),
    )
    fig.tight_layout()
    path = OUT_DIR / "parity_scenario_totals_3models.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------ figure: error by shape


def fig_error_by_shape(shapes: pd.DataFrame) -> Path:
    order = shapes.reindex(shapes["err_pct_xgboost"].abs().sort_values(ascending=False).index)
    order = order.reset_index(drop=True)
    n = len(order)
    clip = 120.0
    h = 0.27
    y0 = np.arange(n)
    fig, ax = plt.subplots(figsize=(11.5, 13.5))
    for i, name in enumerate(order["shape_id"]):
        if name in HIGHLIGHT_SHAPES:
            ax.axhspan(i - 0.5, i + 0.5, color="#fde8e8", zorder=0)
    for k, (fam, lab, col, _) in enumerate(MODELS):
        e = order[f"err_pct_{fam}"].to_numpy()
        ypos = y0 + (k - 1) * h
        ax.barh(ypos, np.clip(e, -clip, clip), height=h * 0.92, color=col, label=lab, zorder=2)
        for yy, v in zip(ypos, e):
            if abs(v) > clip:
                ax.text(np.sign(v) * (clip + 2), yy, f"{v:+.0f}%", va="center",
                        ha="left" if v > 0 else "right", fontsize=7, color=col, fontweight="bold")
    ax.axvline(0, color=SWMM_COLOR, lw=1.0, zorder=3)
    ax.set_yticks(y0)
    labels = ax.set_yticklabels(order["shape_id"], fontsize=8)
    for lbl in labels:
        if lbl.get_text() in HIGHLIGHT_SHAPES:
            lbl.set_color("#b00020")
            lbl.set_fontweight("bold")
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(-clip - 28, clip + 28)
    ax.set_xlabel("Total flood volume error of the shape, predicted vs SWMM (%), "
                  "all factors pooled; positive = over-prediction")
    ax.set_title(f"Per-shape total-volume error of the three models ({SUFFIX})\n"
                 "shapes sorted by |XGBoost error|; red = highlighted shapes; "
                 f"bars clipped at +/-{clip:.0f}% (true value written)",
                 fontsize=FS_TITLE - 0.5, fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    path = OUT_DIR / "error_by_shape.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# -------------------------------------------------------------- figures: maps, profiles


def _scenario_frame(df: pd.DataFrame, shape: str, factor: float) -> pd.DataFrame:
    sub = df[(df["shape_id"] == shape) & (df["factor_mult"] == factor)].copy()
    if len(sub) == 0:
        raise ValueError(f"Empty scenario {shape} f{factor}")
    return sub.sort_values("node_id").reset_index(drop=True)


def _scenario_title(pick: dict) -> str:
    note = ""
    if abs(pick["factor"] - TARGET_FACTOR) > 1e-9:
        note = f" (factor {TARGET_FACTOR:.1f} not available; nearest used)"
    return f"shape {pick['shape']}, factor {pick['factor']:.1f}{note}"


def _scenario_stats(sc: pd.DataFrame) -> dict:
    out = {"n_nodes": int(len(sc)),
           "swmm_total_m3": float(sc["vol_swmm"].sum()),
           "swmm_flooded_nodes": int((sc["vol_swmm"] > 0).sum())}
    for fam, _, _, _ in MODELS:
        p = sc[f"pred_{fam}"].to_numpy()
        y = sc["vol_swmm"].to_numpy()
        out[fam] = {
            "total_m3": float(p.sum()),
            "total_error_pct": float((p.sum() / y.sum() - 1) * 100) if y.sum() > 0 else float("nan"),
            "flooded_nodes": int((p > 0).sum()),
            "false_positive_nodes": int(((p > 0) & (y == 0)).sum()),
            "missed_nodes": int(((p == 0) & (y > 0)).sum()),
            "rmse_nodes_m3": float(np.sqrt(((p - y) ** 2).mean())),
        }
    return out


def fig_flood_map(sc: pd.DataFrame, pick: dict, coords, conduits) -> Path:
    cols = [("vol_swmm", "SWMM (reference)", SWMM_COLOR)] + [
        (f"pred_{f}", f"{lab} (tuned)", c) for f, lab, c, _ in MODELS
    ]
    vmax = max(float(sc[c].max()) for c, _, _ in cols)
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd")
    xy = sc["node_id"].map(lambda n: coords[n])
    xs = np.array([c[0] for c in xy])
    ys = np.array([c[1] for c in xy])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.27, wspace=0.03)
    stats = _scenario_stats(sc)
    mappable = None
    for ax, (col, name, edge) in zip(axes, cols):
        for _lid, a, b in conduits:
            if a in coords and b in coords:
                ax.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]],
                        color="#b0bec5", lw=0.7, zorder=1)
        v = sc[col].to_numpy()
        dry = v <= 0
        ax.scatter(xs[dry], ys[dry], s=10, color="#cfd8dc", edgecolors="#90a4ae",
                   linewidths=0.4, zorder=2)
        wet = ~dry
        sizes = 30 + 220 * np.clip(v[wet] / vmax, 0, 1) ** 0.5
        mappable = ax.scatter(xs[wet], ys[wet], s=sizes, c=v[wet], cmap=cmap, norm=norm,
                              edgecolors=edge, linewidths=0.9, zorder=3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if col == "vol_swmm":
            txt = (f"total {stats['swmm_total_m3']:,.0f} m³\n"
                   f"{stats['swmm_flooded_nodes']} flooded nodes")
        else:
            s = stats[col.replace("pred_", "")]
            txt = (f"total {s['total_m3']:,.0f} m³ ({s['total_error_pct']:+.1f}%)\n"
                   f"{s['flooded_nodes']} flooded nodes "
                   f"(+{s['false_positive_nodes']} false, -{s['missed_nodes']} missed)\n"
                   f"node RMSE {s['rmse_nodes_m3']:.1f} m³")
        ax.set_title(name, fontweight="bold", fontsize=FS_TITLE)
        ax.text(0.5, -0.02, txt, transform=ax.transAxes, ha="center", va="top", fontsize=FS_NOTE)
    cax = fig.add_axes([0.30, 0.075, 0.40, 0.03])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    ticks = [t for t in (0, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000,
                         8000, 10000) if t <= vmax]
    if vmax > 100:  # drop the crowded low ticks on large scales
        ticks = [t for t in ticks if t == 0 or t >= 50]
    cbar.set_ticks(ticks)
    cbar.set_label("Flood volume (m³)  (same colour scale in all four panels; square-root scale)",
                   fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)
    fig.suptitle(f"Flood maps: {_scenario_title(pick)} (selection rule: {pick['rule']})\n"
                 "Out-of-fold predictions, LeaveOneShapeOut; dry nodes in grey; "
                 "marker outline = SWMM grey, XGBoost blue, MLP orange, SVM green",
                 fontsize=FS_TITLE + 1, y=0.97, fontweight="bold")
    path = OUT_DIR / f"flood_map_{pick['rule']}_{pick['shape']}_f{pick['factor']:.1f}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_node_profile(sc: pd.DataFrame, pick: dict) -> Path:
    vis = sc[sc["vol_swmm"] > 0].sort_values(["vol_swmm", "node_id"], ascending=[False, True])
    n = len(vis)
    x = np.arange(n)
    w = 0.2
    fig, ax = plt.subplots(figsize=(min(17, max(11, 0.3 * n)), 6.2))
    ax.bar(x - 1.5 * w, vis["vol_swmm"], w, color=SWMM_COLOR, label="SWMM (reference)")
    for k, (fam, lab, col, _) in enumerate(MODELS):
        ax.bar(x + (k - 0.5) * w, vis[f"pred_{fam}"], w, color=col, label=f"{lab} (tuned)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(vis["node_id"]), rotation=75, ha="right",
                       fontsize=8)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_xlabel("Node ID (nodes flooded in SWMM, sorted by SWMM volume)")
    ax.set_ylabel("Flood volume (m³)")
    st = _scenario_stats(sc)
    fp = ", ".join(f"{LABEL[f]} {st[f]['false_positive_nodes']}" for f, _, _, _ in MODELS)
    ax.set_title(f"Node profile: {_scenario_title(pick)} ({SUFFIX})\n"
                 f"{n} nodes flooded in SWMM; nodes dry in SWMM but predicted flooded: {fp}",
                 fontsize=FS_TITLE - 0.5, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    path = OUT_DIR / f"node_profile_{pick['rule']}_{pick['shape']}_f{pick['factor']:.1f}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# -------------------------------------------------------------------------------- main


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading prepared dataset and out-of-fold predictions...")
    df = load_frame()
    metrics = {fam: pooled_metrics(df, fam) for fam, _, _, _ in MODELS}
    check_against_csv(metrics)
    numbers: dict = {"pooled": metrics}
    numbers["rows_inunda_equals_vol_gt0"] = bool(((df["vol_swmm"] > 0) == (df["inunda"] == 1)).all())

    tot = scenario_totals(df)
    shapes = shape_table(df)
    picks = select_scenarios(df, shapes)
    numbers["selected_scenarios"] = picks
    numbers["shapes"] = shapes.round(4).to_dict(orient="records")

    paths = [fig_parity_nodes(df, metrics), fig_parity_totals(tot, numbers),
             fig_error_by_shape(shapes)]

    coords = parse_coordinates(INP_PATH)
    conduits = parse_conduits(INP_PATH)
    numbers["scenario_stats"] = {}
    for pick in picks:
        sc = _scenario_frame(df, pick["shape"], pick["factor"])
        numbers["scenario_stats"][pick["rule"]] = _scenario_stats(sc)
        paths.append(fig_flood_map(sc, pick, coords, conduits))
        paths.append(fig_node_profile(sc, pick))

    (OUT_DIR / "figure_numbers.json").write_text(
        json.dumps(numbers, indent=1, default=float), encoding="utf-8")
    for p in paths:
        print("wrote", p.relative_to(REPO_ROOT))
    print("Selected scenarios:", [(p["rule"], p["shape"], p["factor"]) for p in picks])


if __name__ == "__main__":
    main()
