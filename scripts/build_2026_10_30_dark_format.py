"""Restyle the 2026-10-30 update to the approved dark deck and point parity plots."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "figures_part2"))
from make_chico_norte_transfer_figures import (  # noqa: E402
    MODEL_NAMES, MODEL_ORDER, OUTPUT_DIR, SOURCE_ROOT, TRANSFER_ROOTS,
    _figure_metrics, _routed_column, _sha256, load_verified_transfer,
)

SOURCE_DECK = ROOT / "Presentacion_MLSMFP_30-10_avance_chico_norte_y_modelos_con_graficas.pptx"
OUTPUT_DECK = ROOT / "Presentacion_MLSMFP_30-10_avance_chico_norte_y_modelos_paridad_global_final.pptx"
FIGURE_DIR = OUTPUT_DIR / "formato_20_10_paridad_global"
PALETTE = {
    "background": "0E2A47",
    "card": "193A58",
    "card_alt": "234560",
    "blue": "5E8CBE",
    "blue_light": "8BC5FA",
    "orange": "F2A65A",
    "gold": "F1CB83",
    "text": "F7FAFE",
    "muted": "B7CBE0",
    "grid": "D9E1EA",
}
LIGHT_FILL_MAP = {
    "E9F2FA": PALETTE["card_alt"],
    "FBF5E8": PALETTE["card"],
    "F2F5F8": PALETTE["card"],
    "F2F6F9": PALETTE["card"],
    "FFFFFF": PALETTE["card"],
}
TEXT_COLOR_MAP = {
    "0E2A47": PALETTE["text"],
    "2D3034": PALETTE["text"],
    "5B6977": PALETTE["muted"],
    "355A84": PALETTE["blue_light"],
    "5E8CBE": PALETTE["blue_light"],
}


def color(hex_string: str) -> RGBColor:
    return RGBColor.from_string(hex_string)


def rgb_of_color(color_obj) -> str | None:
    try:
        if color_obj.type is not None and color_obj.rgb is not None:
            return str(color_obj.rgb).upper()
    except (AttributeError, TypeError, ValueError):
        return None
    return None


def make_parity_figure(frame, path: Path) -> dict:
    plotted = frame
    actual = plotted["vol_inundacion_m3"].to_numpy(dtype=float)
    maximum = max(float(actual.max()), *(float(plotted[_routed_column(m)].max()) for m in MODEL_ORDER), 1.0)
    upper = float(np.log10(1.0 + maximum * 1.05))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, model in zip(axes, MODEL_ORDER):
        pred = plotted[_routed_column(model)].to_numpy(dtype=float)
        x = np.log10(1.0 + actual)
        y = np.log10(1.0 + pred)
        color_hex = PALETTE["blue"] if model == "xgboost" else PALETTE["orange"]
        ax.scatter(x, y, s=3.2, alpha=0.075, color=f"#{color_hex}", linewidths=0, rasterized=True)
        ax.plot([0, upper], [0, upper], linestyle="--", color="#59636E", linewidth=1.1)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color=f"#{PALETTE['grid']}", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_title(MODEL_NAMES[model], fontsize=11, fontweight="bold", color="#0E2A47", pad=8)
        metrics = _figure_metrics(plotted, model)
        sst = float(np.square(actual - actual.mean()).sum())
        r2 = float(1.0 - np.square(pred - actual).sum() / sst) if sst else 0.0
        note = (
            f"RMSE = {metrics['rmse_m3']:.2f} m\u00b3\n"
            f"MAE = {metrics['mae_m3']:.2f} m\u00b3\n"
            f"R\u00b2 = {r2:.3f}  \u00b7  n = {metrics['n']:,}"
        )
        ax.text(0.025, 0.975, note, transform=ax.transAxes, ha="left", va="top", fontsize=8.3,
                color="#202B36", bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C4D1DC", "alpha": 0.96})
        tick_values = [0, 1, 10, 100, 1_000, 10_000, 100_000]
        tick_values = [v for v in tick_values if np.log10(1 + v) <= upper + 1e-9]
        ax.set_xticks([np.log10(1 + v) for v in tick_values])
        ax.set_yticks([np.log10(1 + v) for v in tick_values])
        labels = [f"{v:,}" for v in tick_values]
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.tick_params(colors="#42515F", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#8392A0")
            spine.set_linewidth(0.7)
    axes[0].set_ylabel("Volumen predicho por nodo (m\u00b3)", fontsize=9.2, color="#273746")
    for ax in axes:
        ax.set_xlabel("Volumen SWMM por nodo (m\u00b3)", fontsize=9.2, color="#273746")
    title = "Paridad nodal global \u00b7 410.814 pares de 787 corridas elegibles"
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#0E2A47", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=0.45)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        model: {**_figure_metrics(plotted, model), "r2_descriptive": float(
            1.0 - np.square(plotted[_routed_column(model)].to_numpy(dtype=float) - actual).sum()
            / np.square(actual - actual.mean()).sum()
        ) if np.square(actual - actual.mean()).sum() else 0.0}
        for model in MODEL_ORDER
    }


def replace_picture(slide, image_path: Path) -> None:
    pictures = [shape for shape in slide.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
    if len(pictures) != 1:
        raise ValueError(f"Expected one parity image, found {len(pictures)}")
    old = pictures[0]
    left, top, width, height = old.left, old.top, old.width, old.height
    old._element.getparent().remove(old._element)
    slide.shapes.add_picture(str(image_path), left, top, width=width, height=height)


def set_text(shape, value: str) -> None:
    tf = shape.text_frame
    p0 = tf.paragraphs[0] if tf.paragraphs else None
    align = p0.alignment if p0 else None
    tf.clear()
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    p.add_run().text = value
    tf.word_wrap = True


def repair_text_encoding(text: str) -> str:
    replacements = {
        "evalu? con": "evalu\u00f3 con",
        "Evaluaci?n": "Evaluaci\u00f3n",
        "M?tricas": "M\u00e9tricas",
        "evaluaci?n": "evaluaci\u00f3n",
        "m?tricas": "m\u00e9tricas",
        "l?mites": "l\u00edmites",
        "M?trica": "M\u00e9trica",
        "m?trica": "m\u00e9trica",
        "generalizaci?n": "generalizaci\u00f3n",
        "corrida?nodo": "corrida\u2013nodo",
        "log??(": "log10(",
        "log??": "log10",
        "l?nea": "l\u00ednea",
        "predicci?n": "predicci\u00f3n",
        "auditor?a": "auditor\u00eda",
        "inundaci?n": "inundaci\u00f3n",
        "hidr?ulica": "hidr\u00e1ulica",
        "c?rculos": "c\u00edrculos",
        "Preparaci?n": "Preparaci\u00f3n",
        "? 1 m?": " \u22651 m\u00b3",
        "?1 m?": "\u22651 m\u00b3",
        "m?": "m\u00b3",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace(" ? ", " \u00b7 ")
    if text.startswith("? "):
        text = "\u2022 " + text[2:]
    return text


def repair_presentation_text(prs) -> None:
    for slide in prs.slides:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.text = repair_text_encoding(run.text)
                    if "?" in run.text:
                        raise ValueError(f"Unrepaired question mark in slide text: {shape.name}: {run.text}")


def apply_dark_theme(prs) -> None:
    for slide_index, slide in enumerate(prs.slides):
        if slide_index == 0:
            continue
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = color(PALETTE["background"])
        for shape in slide.shapes:
            old_fill = None
            try:
                if shape.fill.type is not None:
                    old_fill = rgb_of_color(shape.fill.fore_color)
                    if old_fill in LIGHT_FILL_MAP:
                        shape.fill.solid()
                        shape.fill.fore_color.rgb = color(LIGHT_FILL_MAP[old_fill])
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                line_rgb = rgb_of_color(shape.line.color)
                if line_rgb in {"E5EAF0", "D7E0E8", "C9D4DF", "FFFFFF"}:
                    shape.line.color.rgb = color("355A84")
            except (AttributeError, TypeError, ValueError):
                pass
            if not getattr(shape, "has_text_frame", False):
                continue
            text = shape.text.strip()
            if not text:
                continue
            x, y, w, h = [value / 914400 for value in (shape.left, shape.top, shape.width, shape.height)]
            is_title = y < 0.65 and w > 8.0 and h < 0.75
            is_subtitle = 0.60 <= y < 1.05 and w > 8.0
            is_footer = y > 5.15 and x < 8.9
            is_page = y > 5.15 and x >= 8.9 and text.isdigit()
            if is_title:
                target = "blue_light"
            elif is_subtitle or is_footer:
                target = "muted"
            elif is_page:
                target = "blue_light"
            elif old_fill == "8BC5FA" or old_fill == "F1CB83":
                target = "background"
            else:
                target = None
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if is_title:
                        run.font.name = "Libre Baskerville"
                        run.font.size = __import__("pptx").util.Pt(21.5)
                        run.font.bold = True
                    elif is_subtitle:
                        run.font.name = "Montserrat"
                        run.font.size = __import__("pptx").util.Pt(10.0)
                    elif is_footer:
                        run.font.name = "Montserrat"
                        run.font.size = __import__("pptx").util.Pt(8.0)
                    elif is_page:
                        run.font.name = "Montserrat"
                        run.font.size = __import__("pptx").util.Pt(11.0)
                    old_color = rgb_of_color(run.font.color)
                    if target:
                        run.font.color.rgb = color(PALETTE[target])
                    elif old_color in TEXT_COLOR_MAP:
                        run.font.color.rgb = color(TEXT_COLOR_MAP[old_color])
                    elif old_color is None:
                        run.font.color.rgb = color(PALETTE["text"])


def main(*, overwrite: bool = False) -> None:
    if not SOURCE_DECK.is_file():
        raise FileNotFoundError(SOURCE_DECK)
    if OUTPUT_DECK.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite the presentation: {OUTPUT_DECK}")
    if FIGURE_DIR.exists():
        expected = {"paridad_global_puntos.png", "manifest.json", "README.md"}
        existing = {p.name for p in FIGURE_DIR.iterdir()}
        if not existing.issubset(expected):
            raise FileExistsError(f"Figure directory has unexpected contents: {FIGURE_DIR}")
    frame, totals, provenance = load_verified_transfer(TRANSFER_ROOTS, SOURCE_ROOT)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    all_figure = FIGURE_DIR / "paridad_global_puntos.png"
    all_metrics = make_parity_figure(frame, all_figure)

    prs = Presentation(SOURCE_DECK)
    if len(prs.slides) != 18:
        raise ValueError(f"Unexpected source deck slide count: {len(prs.slides)}")
    slide = prs.slides[11]
    set_text(slide.shapes[2], "Paridad nodal global: predicci\u00f3n frente a SWMM")
    set_text(slide.shapes[3], "Cada punto es un par corrida\u2013nodo. La diagonal y = x significa volumen predicho igual a SWMM.")
    set_text(slide.shapes[5], "410.814 pares \u00b7 RMSE global: XGBoost 270,9 m\u00b3; MLP ensemble 435,9 m\u00b3.")
    replace_picture(slide, all_figure)

    for slide_index, level in zip((12, 13, 14), ("bajo", "mediano", "alto")):
        map_slide = prs.slides[slide_index]
        set_text(map_slide.shapes[2], f"Chico Norte: escenario {level}")
        set_text(map_slide.shapes[3], "Mapa de inundaci\u00f3n: comparaci\u00f3n entre SWMM y los modelos.")

    # Remove the conditional parity slide: the deck should show only the global comparison.
    for later_slide in list(prs.slides)[13:]:
        for shape in later_slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text.strip().isdigit():
                set_text(shape, str(int(shape.text.strip()) - 1))
    parity_slide_id = prs.slides._sldIdLst[12]
    prs.part.drop_rel(parity_slide_id.rId)
    prs.slides._sldIdLst.remove(parity_slide_id)

    repair_presentation_text(prs)
    apply_dark_theme(prs)
    prs.save(OUTPUT_DECK)

    data = {
        "schema_version": "chico_norte_20_10_dark_style_global_parity_points_v2",
        "style_reference": "Presentacion_MLSMFP_20-10_avances_auditoria_svm.pptx",
        "source_deck": SOURCE_DECK.name,
        "source_deck_sha256": _sha256(SOURCE_DECK),
        "eligible_runs": provenance["eligible_runs"],
        "excluded_runs": provenance["excluded_runs"],
        "pairs": provenance["rows"],
        "labels_sha256": provenance["labels_sha256"],
        "color_convention": {"xgboost": f"#{PALETTE['blue']}", "mlp": f"#{PALETTE['orange']}"},
        "global_parity": {"figure": all_figure.name, "models": all_metrics, "sha256": _sha256(all_figure)},
        "output_presentation": {"filename": OUTPUT_DECK.name, "sha256": _sha256(OUTPUT_DECK)},
    }
    (FIGURE_DIR / "manifest.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved dark-format deck: {OUTPUT_DECK}")
    print(f"Figures and manifest: {FIGURE_DIR}")
    print(f"Eligible={provenance['eligible_runs']}; rows={provenance['rows']}; labels SHA-256={provenance['labels_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace the generated deck at OUTPUT_DECK.")
    main(overwrite=parser.parse_args().overwrite)
