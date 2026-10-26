"""English hydrograph-shape panels for the base deck (slides 3/4, 6 and 11).

Rebuilt from the real data in ``data/hydrograph_shapes/`` and
``data/hydrograph_validation/`` (not from the original pixels).

Usage (from the repository root):
    venv/Scripts/python.exe scripts/figures_en/hydrograph_panels.py

Writes only to ``docs/figures_en/``:
    slide03_hydrograph_shapes_en.png        (slides 3 and 4)
    slide06_new_training_shapes_en.png      (slide 6)
    slide11_validation_shapes_en.png        (slide 11)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SHAPES_DIR = ROOT / "data" / "hydrograph_shapes"
VALIDATION_DIR = ROOT / "data" / "hydrograph_validation"
OUT_DIR = ROOT / "docs" / "figures_en"

# Output scale: 1 reproduces the pixel size of the original pictures exactly.
SCALE = 1

# --- Slide 3/4: existing (blue) vs. 10 new (orange) training shapes ---------
EXISTING_SHAPES = [
    "common_bogota", "common_storm", "long_moderate", "new_hydrograph",
    "new_hydrograph_2", "rare_extreme", "shape_2h", "shape_5h",
]
NEW_SHAPES_BATCH1 = [
    "shape_back_loaded_late_peak", "shape_double_peak_close",
    "shape_double_peak_intensifying", "shape_flash_flood_urban",
    "shape_gentle_broad_long", "shape_long_recession_tail",
    "shape_slow_broad_rise", "shape_symmetric_moderate",
    "shape_triple_peak_long_decreasing", "shape_triple_peak_short",
]

# --- Slide 6: the 10 newest training shapes (ordered by duration) ----------
NEW_SHAPES_BATCH2 = [
    "shape_micro_burst_45min", "shape_stepped_plateau_1h45",
    "shape_sharp_peak_2h15", "shape_offset_double_peak_2h5",
    "shape_early_broad_6h45", "shape_terminal_peak_7h",
    "shape_five_pulse_8h5", "shape_flashy_long_tail_9h5",
    "shape_sustained_plateau_14h", "shape_twin_plateau_15h",
]

# --- Slide 11: validation shapes (5 interpolation + 2 extrapolation) -------
VALIDATION_SHAPES = [
    ("val_early_flash_2h45", "INTERPOLATION"),
    ("val_single_peak_4h5", "INTERPOLATION"),
    ("val_triple_pulse_5h75", "INTERPOLATION"),
    ("val_double_peak_7h15", "INTERPOLATION"),
    ("val_long_late_11h75", "INTERPOLATION"),
    ("val_extrap_ultrafast_20min", "EXTRAPOLATION"),
    ("val_extrap_marathon_20h", "EXTRAPOLATION"),
]
CATEGORY_COLOR = {"INTERPOLATION": "tab:green", "EXTRAPOLATION": "tab:red"}


def load_shape(folder: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(folder / f"{name}.csv")
    return df[["time_h", "q_norm"]].astype(float)


def duration_and_peak(df: pd.DataFrame) -> tuple[float, float]:
    """Duration = last time stamp; t_peak = time of the (first) maximum."""
    dur = float(df["time_h"].iloc[-1])
    t_peak = float(df["time_h"].iloc[int(df["q_norm"].to_numpy().argmax())])
    return dur, t_peak


def _figure(width_px: int, height_px: int, dpi: int = 100):
    return plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)


def slide03_existing_vs_new() -> Path:
    """5x4 grid; blue = existing shapes, orange = the 10 new shapes."""
    fig = _figure(2340, 2080, dpi=130)
    axes = fig.subplots(5, 4).ravel()
    items = [(n, "tab:blue") for n in EXISTING_SHAPES] + [
        (n, "tab:orange") for n in NEW_SHAPES_BATCH1
    ]
    for ax, (name, color) in zip(axes, items):
        df = load_shape(SHAPES_DIR, name)
        ax.plot(df["time_h"], df["q_norm"], color=color, lw=1.5)
        ax.fill_between(df["time_h"], df["q_norm"], color=color, alpha=0.15)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("t (h)", fontsize=7)
        ax.set_ylabel("q_norm", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes[len(items):]:
        ax.set_visible(False)
    fig.suptitle("Hydrograph shapes — blue: existing, orange: 10 new", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "slide03_hydrograph_shapes_en.png"
    fig.savefig(out, dpi=130 * SCALE)
    plt.close(fig)
    return out


def _detail_panel(ax, df: pd.DataFrame, name: str, tag: str, color: str) -> None:
    dur, t_peak = duration_and_peak(df)
    ax.plot(df["time_h"], df["q_norm"], color=color, lw=2, marker="o", ms=3)
    ax.fill_between(df["time_h"], df["q_norm"], color=color, alpha=0.2)
    ax.axvline(t_peak, color="grey", ls="--", lw=1)
    ax.set_xlim(0, dur)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{name}\ndur={dur:.2f} h  t_peak={t_peak:.2f} h  {tag}", fontsize=8.5)


def slide06_new_training_shapes() -> Path:
    fig = _figure(2172, 782, dpi=110)
    axes = fig.subplots(2, 5).ravel()
    for ax, name in zip(axes, NEW_SHAPES_BATCH2):
        _detail_panel(ax, load_shape(SHAPES_DIR, name), name, "TRAINING", "tab:blue")
    fig.suptitle("10 new TRAINING shapes  (data/hydrograph_shapes/)", fontsize=14)
    fig.supxlabel("time (h)", fontsize=10)
    fig.supylabel("q_norm", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "slide06_new_training_shapes_en.png"
    fig.savefig(out, dpi=110 * SCALE)
    plt.close(fig)
    return out


def slide11_validation_shapes() -> Path:
    fig = _figure(1790, 782, dpi=110)
    axes = fig.subplots(2, 4).ravel()
    for ax, (name, tag) in zip(axes, VALIDATION_SHAPES):
        _detail_panel(ax, load_shape(VALIDATION_DIR, name), name, tag, CATEGORY_COLOR[tag])
    for ax in axes[len(VALIDATION_SHAPES):]:
        ax.set_visible(False)
    fig.suptitle(
        "7 VALIDATION shapes  (data/hydrograph_validation/) — "
        "green = interpolation, red = extrapolation",
        fontsize=14,
    )
    fig.supxlabel("time (h)", fontsize=10)
    fig.supylabel("q_norm", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "slide11_validation_shapes_en.png"
    fig.savefig(out, dpi=110 * SCALE)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for build in (slide03_existing_vs_new, slide06_new_training_shapes, slide11_validation_shapes):
        print("wrote", build().relative_to(ROOT))
    # Print the recomputed annotations so they can be checked against the originals.
    for folder, names in (
        (SHAPES_DIR, NEW_SHAPES_BATCH2),
        (VALIDATION_DIR, [n for n, _ in VALIDATION_SHAPES]),
    ):
        for name in names:
            dur, t_peak = duration_and_peak(load_shape(folder, name))
            print(f"  {name}: dur={dur:.2f} h  t_peak={t_peak:.2f} h")


if __name__ == "__main__":
    main()
