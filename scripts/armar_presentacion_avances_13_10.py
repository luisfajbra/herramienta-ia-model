"""Build the October 13, 2026 presentation (progress: audit, approximate SVM
and fair comparison) from the October 5 presentation.

The base presentation is never modified: the script opens it, translates its
text to English (in the output only, driven by deck_translations_en.json),
appends 6 new slides (plus 4 results slides when the part 2 tables exist) and
saves the result to a different file. It is idempotent: the output is always
rebuilt from the base.

Figures on the audit slides come from the CSVs in
``docs/auditoria/correcciones/datos`` (read at build time); the rest is text
from ``docs/auditoria/correcciones/0[1-4]-*.md`` and from the spec
``docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md``.
CSV column names and file names are code identifiers and stay in Spanish.

Usage:
    venv/Scripts/python.exe scripts/armar_presentacion_avances_13_10.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import (  # noqa: E402
    XL_CHART_TYPE,
    XL_MARKER_STYLE,
    XL_LABEL_POSITION,
    XL_LEGEND_POSITION,
    XL_TICK_MARK,
)
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Pt  # noqa: E402

BASE = REPO_ROOT / "Presentacion_MLSMFP_05-10_resultados_fase_c2.pptx"
OUTPUT = REPO_ROOT / "Presentacion_MLSMFP_20-10_avances_auditoria_svm.pptx"
OUTPUT_LEGACY = REPO_ROOT / "Presentacion_MLSMFP_13-10_avances_auditoria_svm.pptx"
DATOS_DIR = REPO_ROOT / "docs" / "auditoria" / "correcciones" / "datos"
DATOS_PARTE2_DIR = REPO_ROOT / "docs" / "auditoria" / "correcciones" / "datos_parte2"
FIGURAS_DIR = REPO_ROOT / "docs" / "figures_part2"
FIGURAS_EN_DIR = REPO_ROOT / "docs" / "figures_en"

FOOTER_TEXT = (
    "MLSMFP  |  Progress: audit, approximate SVM and fair comparison  |  October 2026"
)

# --- Estilo de la presentacion base (colores, tipografias y posiciones) -----
BG = "0E2A47"
PANEL = "193A58"
PANEL_ALT = "234560"
HEAD = "355A84"
LINE = "5E8CBE"
ACCENT = "8BC5FA"
TEXT = "F7FAFE"
MUTED = "B7CBE0"
AMBER = "F1CB83"
GREEN = "A4FF9F"
FONT = "Montserrat"
FONT_TITLE = "Libre Baskerville"

SLIDE_W = 9144000
SLIDE_H = 5143500
L = 292608  # margen izquierdo de las tablas de la base
W = 8549640  # ancho util de las tablas de la base
Y0 = 1030000  # inicio del area de contenido
LAYOUT_NAME = "Title only"


# ---------------------------------------------------------------------------
# Formato y lectura de datos
# ---------------------------------------------------------------------------
def fmt(value: float, decimals: int = 2) -> str:
    """English format: decimal point and thousands comma (106526 -> 106,526)."""
    return f"{float(value):,.{decimals}f}"


def _rel(ruta) -> str:
    """Path relative to the repository (for speaker notes), POSIX style."""
    ruta = Path(ruta)
    try:
        return ruta.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return ruta.as_posix()


def cargar_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fila(rows: list[dict], modelo: str, protocolo: str) -> dict:
    for row in rows:
        if row["modelo"] == modelo and row["protocolo"] == protocolo:
            return row
    raise KeyError(f"No row for {modelo} / {protocolo}")


# ---------------------------------------------------------------------------
# Primitivas de dibujo
# ---------------------------------------------------------------------------
def _rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def P(text="", size=14, bold=False, color=TEXT, after=0, runs=None) -> dict:
    """Describe un parrafo; `runs` = [(texto, color, negrita)] opcional."""
    return {"text": text, "size": size, "bold": bold, "color": color,
            "after": after, "runs": runs}


def _fill_text_frame(tf, paras, anchor=MSO_ANCHOR.TOP, margins=(0, 0, 0, 0)) -> None:
    tf.word_wrap = True
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (Emu(m) for m in margins)
    tf.vertical_anchor = anchor
    first = True
    for spec in paras:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = PP_ALIGN.LEFT
        para.line_spacing = 1.0
        para.space_before = Pt(0)
        para.space_after = Pt(spec["after"])
        runs = spec["runs"] or [(spec["text"], spec["color"], spec["bold"])]
        for text, color, bold in runs:
            run = para.add_run()
            run.text = text
            run.font.size = Pt(spec["size"])
            run.font.bold = bold
            run.font.name = FONT
            run.font.color.rgb = _rgb(color)


def _textbox(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, name=None):
    box = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    if name:
        box.name = name
    _fill_text_frame(box.text_frame, paras, anchor)
    return box


def _shape(slide, kind, x, y, w, h, fill=None, line=None, line_w=1.0, paras=None,
           anchor=MSO_ANCHOR.MIDDLE, margins=(91440, 45720, 91440, 45720),
           radius=0.10, name=None):
    shp = slide.shapes.add_shape(kind, Emu(x), Emu(y), Emu(w), Emu(h))
    style = shp._element.find(qn("p:style"))
    if style is not None:  # sin sombras ni efectos del tema
        shp._element.remove(style)
    if name:
        shp.name = name
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = _rgb(line)
        shp.line.width = Pt(line_w)
    else:
        shp.line.fill.background()
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    _fill_text_frame(shp.text_frame, paras or [P("")], anchor, margins)
    return shp


def _card(slide, x, y, w, h, paras, fill=PANEL, line=LINE, line_w=1.0,
          anchor=MSO_ANCHOR.MIDDLE):
    return _shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, line,
                  line_w, paras, anchor, margins=(110000, 60000, 110000, 60000))


def _arrow(slide, x, y, w, h, direction="right", color=ACCENT):
    kind = MSO_SHAPE.RIGHT_ARROW if direction == "right" else MSO_SHAPE.DOWN_ARROW
    return _shape(slide, kind, x, y, w, h, fill=color, line=None)


def _pill(slide, x, y, w, h, text, fill, color=BG, size=10):
    pill = _shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, None,
                  paras=[P(text, size, True, color)], radius=0.5,
                  margins=(110000, 0, 110000, 0))
    pill.text_frame.word_wrap = False
    return pill


def _set_cell(cell, value, size, bold, color, fill) -> None:
    cell.margin_top = cell.margin_bottom = Emu(45720)
    cell.margin_left = cell.margin_right = Emu(91440)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.fill.solid()
    cell.fill.fore_color.rgb = _rgb(fill)
    tf = cell.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.LEFT
    para.line_spacing = 1.0
    para.space_before = Pt(0)
    para.space_after = Pt(0)
    runs = value if isinstance(value, list) else [(value, color, bold)]
    for text, run_color, run_bold in runs:
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = run_bold
        run.font.name = FONT
        run.font.color.rgb = _rgb(run_color)


def _tabla(slide, x, y, col_widths, header, rows, row_h, header_h, size=12,
           name="Table 7"):
    n_rows, n_cols = len(rows) + 1, len(col_widths)
    frame = slide.shapes.add_table(n_rows, n_cols, Emu(x), Emu(y),
                                   Emu(sum(col_widths)), Emu(header_h + row_h * len(rows)))
    frame.name = name
    table = frame.table
    for idx, width in enumerate(col_widths):
        table.columns[idx].width = Emu(width)
    table.rows[0].height = Emu(header_h)
    for idx, text in enumerate(header):
        _set_cell(table.cell(0, idx), text, size, True, TEXT, HEAD)
    for r_idx, row in enumerate(rows, start=1):
        table.rows[r_idx].height = Emu(row_h)
        fill = PANEL if r_idx % 2 == 1 else PANEL_ALT
        for c_idx, value in enumerate(row):
            _set_cell(table.cell(r_idx, c_idx), value, size, False, TEXT, fill)
    return frame


def _chart_barras(slide, x, y, w, h, categorias, series, *, horizontal=False,
                  etiquetas=None, colores_puntos=None, leyenda=True, size=12,
                  gap=60, maximo=None):
    """Grafico de barras agrupadas nativo.

    series = [(nombre, valores, color)]; etiquetas[s][i] = texto de la etiqueta
    (asi la coma decimal no depende de la configuracion regional);
    colores_puntos[s][i] = color de cada barra (para series de un solo color
    por categoria).
    """
    data = CategoryChartData()
    data.categories = categorias
    for nombre, valores, _ in series:
        data.add_series(nombre, valores)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    frame = slide.shapes.add_chart(kind, Emu(x), Emu(y), Emu(w), Emu(h), data)
    chart = frame.chart
    chart.has_title = False
    chart.font.size = Pt(size)
    chart.font.name = FONT
    chart.font.color.rgb = _rgb(TEXT)
    chart.has_legend = leyenda and len(series) > 1
    if chart.has_legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
        chart.legend.font.name = FONT
        chart.legend.font.color.rgb = _rgb(TEXT)
    plot = chart.plots[0]
    plot.gap_width = gap
    plot.vary_by_categories = False
    if len(series) > 1:
        plot.overlap = -8
    value_axis = chart.value_axis
    value_axis.visible = False
    value_axis.has_major_gridlines = False
    value_axis.minimum_scale = 0
    top = maximo or max(max(vals) for _, vals, _ in series) * (1.4 if horizontal else 1.22)
    value_axis.maximum_scale = top
    cat_axis = chart.category_axis
    cat_axis.has_major_gridlines = False
    cat_axis.major_tick_mark = XL_TICK_MARK.NONE
    cat_axis.format.line.color.rgb = _rgb(LINE)
    cat_axis.tick_labels.font.size = Pt(size)
    cat_axis.tick_labels.font.name = FONT
    cat_axis.tick_labels.font.color.rgb = _rgb(TEXT)
    if horizontal:
        cat_axis.reverse_order = True
    for s_idx, (series_obj, (_, valores, color)) in enumerate(zip(plot.series, series)):
        series_obj.format.fill.solid()
        series_obj.format.fill.fore_color.rgb = _rgb(color)
        series_obj.invert_if_negative = False
        for p_idx in range(len(valores)):
            point = series_obj.points[p_idx]
            if colores_puntos:
                point.format.fill.solid()
                point.format.fill.fore_color.rgb = _rgb(colores_puntos[s_idx][p_idx])
            if etiquetas and etiquetas[s_idx] is not None:
                label = point.data_label
                label.position = XL_LABEL_POSITION.OUTSIDE_END
                frame_text = label.text_frame
                frame_text.text = etiquetas[s_idx][p_idx]
                run = frame_text.paragraphs[0].runs[0]
                run.font.size = Pt(size + 1)
                run.font.bold = True
                run.font.name = FONT
                run.font.color.rgb = _rgb(TEXT)
    return frame


# ---------------------------------------------------------------------------
# Diapositiva base (misma "chrome" que las slides 17-37 de la presentacion)
# ---------------------------------------------------------------------------
def _chrome_textbox(slide, name, x, y, w, h, text, size, bold, color, font):
    box = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = Emu(0)
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.LEFT
    para.line_spacing = 1.06
    para.space_before = Pt(0)
    para.space_after = Pt(3)
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = font
    run.font.color.rgb = _rgb(color)
    return box


def _layout(prs):
    for layout in prs.slide_layouts:
        if layout.name == LAYOUT_NAME:
            return layout
    return prs.slide_layouts[len(prs.slide_layouts) - 1]


def nueva_slide(prs, titulo: str, subtitulo: str, pie: str = FOOTER_TEXT):
    """Agrega una slide con el estilo de la base: fondo, barra, titulo,
    subtitulo, pie y numero de pagina (continuando la numeracion)."""
    slide = prs.slides.add_slide(_layout(prs))
    for placeholder in list(slide.shapes):
        placeholder._element.getparent().remove(placeholder._element)
    numero = len(prs.slides)

    def _rect(name, x, y, w, h, color):
        rect = _shape(slide, MSO_SHAPE.RECTANGLE, x, y, w, h, fill=color, name=name)
        return rect

    _rect("Rectangle 1", 0, 0, SLIDE_W, SLIDE_H, BG)
    _rect("Rectangle 2", 274320, 210312, 50292, 411480, ACCENT)
    _chrome_textbox(slide, "TextBox 3", 429768, 192024, 8339327, 393192,
                    titulo, 22, True, ACCENT, FONT_TITLE)
    _chrome_textbox(slide, "TextBox 4", 429768, 658368, 8275320, 329184,
                    subtitulo, 10.5, False, MUTED, FONT)
    _chrome_textbox(slide, "TextBox 5", 310896, 4892040, 7863840, 137160,
                    pie, 8, False, MUTED, FONT)
    _chrome_textbox(slide, "TextBox 6", 8321040, 4818888, 502920, 228600,
                    str(numero), 11, False, ACCENT, FONT)
    return slide


def _notas(slide, texto: str) -> None:
    slide.notes_slide.notes_text_frame.text = texto


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Slide 1: audit
# ---------------------------------------------------------------------------
def slide_auditoria(prs, datos_dir: Path = DATOS_DIR):
    cp = cargar_csv(Path(datos_dir) / "comparacion_protocolos.csv")
    xgb_loso = _fila(cp, "xgboost", "LOSO")
    ref_loso = _fila(cp, "baseline_interpolacion_factor", "LOSO")

    s = nueva_slide(
        prs, "Audit: what was found and what was fixed",
        "Seven findings from the October 10 audit · default XGBoost, 154,080 rows",
    )

    def cod(codigo, resto):
        return [(codigo + "  ", ACCENT, True), (resto, TEXT, False)]

    estado = {"Fixed": GREEN, "Partial": AMBER, "Open": AMBER, "Pending": AMBER}
    filas_raw = [
        (cod("M-01", "No baselines"),
         "Two no-learning baselines added to the benchmark",
         f"LOSO: baseline {fmt(ref_loso['rmse_vol_agregado_m3'])} m³ vs XGBoost "
         f"{fmt(xgb_loso['rmse_vol_agregado_m3'])} m³", "Fixed"),
        (cod("M-02", "Shape volume"),
         "Per-shape measurement; the model is unchanged",
         "One shape = 71.6% of the squared error", "Open"),
        (cod("M-03", "Averaged F1"),
         "Pooled F1 from the confusion matrix",
         f"LOSO: F1 {fmt(xgb_loso['f1_media_por_fold_con_ceros'], 4)} → "
         f"{fmt(xgb_loso['f1_agregado'], 4)}", "Fixed"),
        (cod("V-01", "In-sample figures"),
         "“In-sample” footnote on the new figures",
         "6 tests; old figures not regenerated", "Partial"),
        (cod("V-07", "Script overwrote the .inp"),
         "No side effects on import; requires --force",
         "Does not overwrite without ‑‑force (tests)", "Fixed"),
        (cod("M-07 / V-04", "Model ≠ promotion.json"),
         "predict_network checks the SHA-256 hashes",
         "Stops if they differ (15 tests)", "Partial"),
        (cod("D-04", "Label resolution"),
         "Spike: effective threshold ≈ 0.5 m³",
         "A/B decision pending; I recommend A", "Pending"),
    ]
    filas = [[a, b, c, [(d, estado[d], True)]] for a, b, c, d in filas_raw]
    _tabla(s, L, Y0, [2250000, 2400000, 2650000, 1250000],
           ["Finding", "Fix", "Verification", "Status"], filas,
           row_h=480000, header_h=340000, size=12)

    _notas(s, (
        "The internal audit of the evaluation protocol found seven problems that affected how the results are read; "
        "this table summarizes each finding, its fix, and how it was verified. "
        "The most important is M-01: in LOSO a baseline that only interpolates between neighboring factors reaches a volume RMSE of "
        f"{fmt(ref_loso['rmse_vol_agregado_m3'])} m³ against {fmt(xgb_loso['rmse_vol_agregado_m3'])} m³ for XGBoost, so that protocol measures interpolation, not generalization. "
        f"M-03 is about aggregation: averaging F1 per fold with zeros gave {fmt(xgb_loso['f1_media_por_fold_con_ceros'], 4)} while the pooled F1 is {fmt(xgb_loso['f1_agregado'], 4)}. "
        "Two items are not closed: M-02 remains open (a single shape concentrates 71.6% of the squared error) and D-04 awaits my decision between paths A and B. "
        "V-01 and M-07/V-04 are fixed only for the new figures and for predict_network; other loaders still lack verification.\n\n"
        "Sources: docs/auditoria/correcciones/04-informe-consolidado.md (§1 status per finding, §2 before and after); "
        "datos/comparacion_protocolos.csv (rmse_vol_agregado_m3, f1_agregado, f1_media_por_fold_con_ceros, XGBoost and "
        "baseline_interpolacion_factor in LOSO); datos/por_forma.csv (71.6%: rmse_vol^2 x n_rows of shape_quasi_uniform_block over the total, XGBoost, LeaveOneShapeOut); "
        "01-evaluacion-honesta.md §1-§3; 02-trazabilidad-y-seguridad.md §1-§3 (6 tests for V-01, 15 for verify_promotion); 03-resolucion-de-la-etiqueta.md §1."
    ))
    return s


# ---------------------------------------------------------------------------
# Slide 2: what each protocol measures
# ---------------------------------------------------------------------------
def slide_protocolos(prs, datos_dir: Path = DATOS_DIR):
    cp = cargar_csv(Path(datos_dir) / "comparacion_protocolos.csv")
    ref_loso = _fila(cp, "baseline_interpolacion_factor", "LOSO")
    xgb_loso = _fila(cp, "xgboost", "LOSO")
    ref_forma = _fila(cp, "baseline_forma_mas_cercana", "LeaveOneShapeOut")
    xgb_forma = _fila(cp, "xgboost", "LeaveOneShapeOut")

    s = nueva_slide(
        prs, "What each protocol measures (and why it matters)",
        "Pooled volume RMSE: default XGBoost vs a baseline that does not learn",
    )
    card_w = 3950000
    _card(s, L, Y0 + 20000, card_w, 1330000, [
        P("LOSO and GroupKFold5", 14, True, TEXT, 3),
        P("Intensity factors are held out", 12, False, MUTED, 5),
        P("Interpolation between factors", 14, True, AMBER, 3),
        P("All shapes were already seen", 12, False, MUTED),
    ])
    _card(s, L, Y0 + 1480000, card_w, 1330000, [
        P("LeaveOneShapeOut", 14, True, TEXT, 3),
        P("One whole hydrograph shape is held out", 12, False, MUTED, 5),
        P("Unseen hydrograph shapes", 14, True, ACCENT, 3),
        P("40 folds: the main test", 12, False, MUTED),
    ], fill=PANEL_ALT, line=ACCENT, line_w=2.25)

    cx = L + 4150000
    cw = W - 4150000
    _textbox(s, cx, Y0, cw, 260000,
             [P("Pooled volume RMSE (m³): lower is better", 11, False, MUTED)])
    ref_loso_v = round(float(ref_loso["rmse_vol_agregado_m3"]), 2)
    xgb_loso_v = round(float(xgb_loso["rmse_vol_agregado_m3"]), 2)
    ref_forma_v = round(float(ref_forma["rmse_vol_agregado_m3"]), 2)
    xgb_forma_v = round(float(xgb_forma["rmse_vol_agregado_m3"]), 2)
    _chart_barras(
        s, cx, Y0 + 250000, cw, 2560000,
        ["LOSO", "LeaveOneShapeOut"],
        [("No-learning baseline", [ref_loso_v, ref_forma_v], AMBER),
         ("XGBoost", [xgb_loso_v, xgb_forma_v], ACCENT)],
        etiquetas=[[fmt(ref_loso_v), fmt(ref_forma_v)], [fmt(xgb_loso_v), fmt(xgb_forma_v)]],
        size=12, gap=55,
    )
    _textbox(s, cx, Y0 + 2800000, cw, 200000, [P(
        "Baseline: factor interpolation (left), nearest shape (right)",
        10, False, MUTED)])

    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, 4080000, W, 650000, fill=HEAD, line=None,
           paras=[P("LeaveOneShapeOut is the headline figure", 18, True, TEXT, 2),
                  P("In LOSO the no-learning baseline wins: that protocol only measures interpolation",
                    12, False, MUTED)],
           margins=(150000, 40000, 150000, 40000))

    _notas(s, (
        "Each protocol holds out a different group. In LOSO and GroupKFold5 intensity factors are held out and all shapes remain in training, "
        "so what is measured is interpolation between factors; in LeaveOneShapeOut a whole shape is held out and what is measured is generalization to unseen shapes. "
        f"The proof that this matters: in LOSO the no-learning baseline has an RMSE of {fmt(ref_loso_v)} m³ and XGBoost {fmt(xgb_loso_v)} m³, so winning in LOSO proves nothing. "
        f"In LeaveOneShapeOut the order reverses: the nearest-shape baseline has {fmt(ref_forma_v)} m³ and XGBoost {fmt(xgb_forma_v)} m³. "
        "That is why LeaveOneShapeOut is the headline figure of the whole comparison. With the 25-factor grid, GroupKFold5 is no more demanding than LOSO: no fold holds out two contiguous factors.\n\n"
        "Sources: docs/auditoria/correcciones/datos/comparacion_protocolos.csv, column rmse_vol_agregado_m3 "
        "(baseline_interpolacion_factor/LOSO, xgboost/LOSO, baseline_forma_mas_cercana/LeaveOneShapeOut, xgboost/LeaveOneShapeOut); "
        "01-evaluacion-honesta.md §0 (what each protocol measures) and §1; 04-informe-consolidado.md §3 and §11."
    ))
    return s


# ---------------------------------------------------------------------------
# Slide 3: confusion matrix
# ---------------------------------------------------------------------------
def slide_matriz(prs, datos_dir: Path = DATOS_DIR):
    datos_dir = Path(datos_dir)
    mc = _fila(cargar_csv(datos_dir / "matriz_confusion.csv"), "xgboost", "LeaveOneShapeOut")
    cp = _fila(cargar_csv(datos_dir / "comparacion_protocolos.csv"), "xgboost", "LeaveOneShapeOut")
    bl = _fila(cargar_csv(datos_dir / "banda_limitrofe.csv"), "xgboost", "LeaveOneShapeOut")
    tn, fp, fn, tp = (int(mc[k]) for k in ("tn", "fp", "fn", "tp"))
    fn_banda, fn_total = int(bl["fn_en_banda_1_2m3"]), int(bl["fn_total"])

    s = nueva_slide(
        prs, "XGBoost confusion matrix on unseen shapes",
        f"LeaveOneShapeOut · {fmt(tn + fp + fn + tp, 0)} node-scenarios · label: volume ≥ 1 m³",
    )
    rh_w, cell_w, cell_h, gap = 1250000, 1900000, 980000, 50000
    mx = L + rh_w + 50000
    my1, my2 = 1500000, 1500000 + cell_h + gap
    for i, texto in enumerate(("no flood", "flood")):
        _textbox(s, mx + i * (cell_w + gap), Y0 + 20000, cell_w, 400000,
                 [P("PREDICTED", 10, True, MUTED), P(texto, 14, True, TEXT)])
    for texto, y in (("no flood", my1), ("flood", my2)):
        _textbox(s, L, y, rh_w, cell_h, [P("ACTUAL", 10, True, MUTED), P(texto, 14, True, TEXT)],
                 anchor=MSO_ANCHOR.MIDDLE)

    def celda(x, y, valor, etiqueta, fill, line, line_w, color_num):
        _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, cell_w, cell_h, fill, line, line_w,
               paras=[P(fmt(valor, 0), 24, True, color_num, 2), P(etiqueta, 12, False, MUTED)],
               anchor=MSO_ANCHOR.TOP, margins=(130000, 90000, 100000, 40000))

    celda(mx, my1, tn, "TN · true negatives", PANEL_ALT, LINE, 1.0, TEXT)
    celda(mx + cell_w + gap, my1, fp, "FP · false positives", PANEL, AMBER, 1.0, AMBER)
    celda(mx, my2, fn, "FN · false negatives", PANEL, AMBER, 3.0, AMBER)
    celda(mx + cell_w + gap, my2, tp, "TP · true positives", HEAD, ACCENT, 1.5, TEXT)

    tx = mx + 2 * cell_w + gap + 250000
    tw = L + W - tx
    tile_h = 780000
    metricas = [("Recall", fmt(cp["recall_agregado"], 4), ACCENT),
                ("Precision", fmt(cp["precision_agregada"], 4), ACCENT),
                ("Pooled F1", fmt(cp["f1_agregado"], 4), TEXT)]
    for i, (nombre, valor, color) in enumerate(metricas):
        _card(s, tx, Y0 + 20000 + i * (tile_h + 60000), tw, tile_h,
              [P(nombre, 12, False, MUTED, 1), P(valor, 22, True, color)],
              fill=PANEL, line=LINE)

    cy, ch = 3680000, 950000
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, cy, W, ch, fill=PANEL, line=AMBER, line_w=2.0)
    _textbox(s, L + 150000, cy, 2900000, ch,
             [P(f"{fn_banda} of {fn_total}", 30, True, AMBER)], anchor=MSO_ANCHOR.MIDDLE)
    _textbox(s, L + 3150000, cy, W - 3300000, ch, [
        P("false negatives have a reported volume of 1 m³", 14, True, TEXT, 3),
        P("the minimum that counts as flooding; none exceeds 100 m³", 12, False, MUTED),
    ], anchor=MSO_ANCHOR.MIDDLE)

    _notas(s, (
        "This is the pooled confusion matrix of default XGBoost when a whole hydrograph shape is held out, over "
        f"{fmt(tn + fp + fn + tp, 0)} node-scenario cases. There are {fmt(tp, 0)} true positives, {fmt(tn, 0)} true negatives, {fmt(fp, 0)} false positives, and {fmt(fn, 0)} false negatives; "
        f"from these come a recall of {fmt(cp['recall_agregado'], 4)}, a precision of {fmt(cp['precision_agregada'], 4)}, and an F1 of {fmt(cp['f1_agregado'], 4)}. "
        f"The takeaway: {fn_banda} of the {fn_total} false negatives are nodes whose reported volume is 1 m³, the minimum the label counts as flooding; "
        "none exceeds 100 m³. In other words, the classification errors concentrate at the edge of the label and not on large floods. "
        "Excluding that band raises F1 from 0.9817 to 0.9847, but that is reported only as a sensitivity check, with asymmetric exclusion.\n\n"
        "Sources: docs/auditoria/correcciones/datos/matriz_confusion.csv (xgboost, LeaveOneShapeOut: tn, fp, fn, tp); "
        "datos/comparacion_protocolos.csv (recall_agregado, precision_agregada, f1_agregado); "
        "datos/banda_limitrofe.csv (fn_total, fn_en_banda_1_2m3, fn_con_vol_ge_100m3); 04-informe-consolidado.md §3 and §4; 03-resolucion-de-la-etiqueta.md §1 point 6 (F1 0.9847 without the band)."
    ))
    return s


# ---------------------------------------------------------------------------
# Slide 4: label and resolution
# ---------------------------------------------------------------------------
def slide_etiqueta(prs):
    s = nueva_slide(
        prs, "The flood label and its resolution",
        "From SWMM to the .rpt to the label: the report prints volume at 1 m³ resolution",
    )
    bw, gap_w = 2500000, 524820
    y, h = Y0 + 20000, 1000000
    cajas = [
        ("1 · SWMM", "Continuous overflow volume (floating point)"),
        ("2 · .rpt report", "10⁶ L with 3 decimals: 1 m³ resolution"),
        ("3 · Label", "flood = 1 if V_rpt ≥ 1 m³"),
    ]
    for i, (titulo, detalle) in enumerate(cajas):
        x = L + i * (bw + gap_w)
        _card(s, x, y, bw, h, [P(titulo, 14, True, ACCENT, 4), P(detalle, 13, False, TEXT)],
              fill=PANEL_ALT if i == 1 else PANEL, line=AMBER if i == 1 else LINE,
              line_w=2.0 if i == 1 else 1.0, anchor=MSO_ANCHOR.TOP)
        if i < 2:
            _arrow(s, x + bw + 70000, y + h // 2 - 150000, gap_w - 140000, 300000)

    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, 2200000, W, 800000, fill=HEAD, line=None,
           paras=[P("label ≈ continuous volume ≥ 0.5 m³", 18, True, TEXT, 3),
                  P("bounded to 0.4988–0.5002 m³ in 15 control scenarios (2,400 node-scenario cases)",
                    12, False, MUTED)],
           margins=(150000, 40000, 150000, 40000))

    _textbox(s, L, 3110000, W, 200000,
             [P("PENDING DECISION (AUTHOR)", 10, True, MUTED)])
    cw = (W - 150000) // 2
    cy, ch = 3450000, 1150000
    _card(s, L, cy, cw, ch, [
        P("A · State the assumption", 14, True, ACCENT, 4),
        P("Keep the data and report the confusion matrix with the 1 m³ band separated", 12, False, TEXT)],
        fill=PANEL_ALT, line=AMBER, line_w=2.25)
    _pill(s, L + cw - 2150000, cy - 110000, 2050000, 240000, "RECOMMENDATION: A", AMBER)
    _card(s, L + cw + 150000, cy, cw, ch, [
        P("B · Re-simulate the 1000 runs", 14, True, ACCENT, 4),
        P("≈ 17–20 min of SWMM, but the dataset, DB, benchmark, Optuna and all figures are redone",
          12, False, TEXT)])

    _notas(s, (
        "The flood label does not come directly from SWMM's continuous volume: the .rpt report prints it in 10⁶ L with three decimals, that is, at 1 m³ resolution, "
        "and the label is reported volume greater than or equal to 1 m³. In the control re-simulations this is equivalent to a continuous-volume threshold of about 0.5 m³, "
        "bounded between 0.4988 and 0.5002 m³ in 15 scenarios with per-node records, with no discrepancies in 2,400 node-scenario cases; it is a deterministic quantization, not noise. "
        "There are two paths: A, keep the resolution and threshold, state them as an assumption, and report the confusion matrix with the band separated; or B, re-simulate the 1000 runs, which costs about 17 to 20 minutes of SWMM "
        "but forces redoing the dataset, database, prep_id, benchmark, Optuna, and all figures. The recommendation is path A; the decision is mine and is still pending.\n\n"
        "Sources: docs/auditoria/correcciones/03-resolucion-de-la-etiqueta.md §1 (points 1-8) and §2.1 (resolution chain); "
        "04-informe-consolidado.md §4. The 0.4988-0.5002 m³ bound and the 2,400 cases: 03 §1 point 2; 17-20 min: 03 §1 point 5. "
        "That the decision remains open is also in 04 §11 (Pending)."
    ))
    return s


# ---------------------------------------------------------------------------
# Slide 5: SVM architecture
# ---------------------------------------------------------------------------
def _n_trials(datos_parte2_dir) -> str:
    """Optuna budget (reads resumen_afinacion.csv when available)."""
    if datos_parte2_dir:
        ruta = Path(datos_parte2_dir) / "resumen_afinacion.csv"
        if ruta.exists():
            ns = {r["n_trials"] for r in cargar_csv(ruta)}
            if len(ns) == 1:
                return f"{ns.pop()} Optuna trials per family"
    return "Same number of Optuna trials"


def _tiempo_estudio_svm(datos_parte2_dir) -> dict:
    """Measured cost of the approximate SVM. The per-fold time (8-13 s with
    ~150,000 rows: default svm 8 s; n_components 400, 13 s) comes from the
    cost study; the study duration comes from resumen_afinacion.csv."""
    if datos_parte2_dir:
        ruta = Path(datos_parte2_dir) / "resumen_afinacion.csv"
        if ruta.exists():
            for r in cargar_csv(ruta):
                if r["family"] == "svm":
                    minutos = fmt(float(r["duration_s"]) / 60, 0)
                    return {
                        "slide": "Linear cost in n: ~8–13 s per fold on 150,000 rows; "
                                 f"{r['n_trials']}-trial study in ~{minutos} min",
                        "notas": "fit and predict take ~8 to 13 s per fold on 150,000 rows "
                                 "(default SVM 8 s; n_components 400, 13 s) and the full "
                                 f"{r['n_trials']}-trial study took about {minutos} minutes.",
                        "fuente": f"{_rel(datos_parte2_dir)}/resumen_afinacion.csv "
                                  "(family=svm, duration_s) and the part 2 cost study.",
                    }
    return {
        "slide": "Linear cost in n; expected to run in seconds per fold (to be measured)",
        "notas": "the actual per-fold time is still to be measured; today it is an expectation, not a figure.",
        "fuente": "per-fold time pending measurement.",
    }


def slide_svm(prs, datos_parte2_dir=None):
    medido = _tiempo_estudio_svm(datos_parte2_dir)
    s = nueva_slide(
        prs, "Approximate SVM: architecture change",
        "The exact RBF kernel does not scale to 154,080 rows; it is approximated with a Nystroem map",
    )
    gap = 190000
    x1, w1 = L, 1000000
    x2, w2 = x1 + w1 + gap, 1900000
    x3, w3 = x2 + w2 + gap, 2250000
    x4, w4 = x3 + w3 + gap, L + W - (x3 + w3 + gap)

    # Before
    ya, ha = 1320000, 800000
    _pill(s, L, 1030000, 2400000, 250000, "BEFORE · EXACT KERNEL", AMBER)
    _card(s, x1, ya, w1, ha, [P("Data", 14, True, TEXT)])
    _card(s, x2, ya, w2, ha, [P("StandardScaler", 14, True, TEXT)])
    _card(s, x3, ya, w3 + gap + w4, ha, [
        P("SVC / SVR with exact RBF kernel", 14, True, TEXT, 3),
        P("Cost O(n²) to O(n³) with ~154,000 rows", 12, False, MUTED)],
        fill=PANEL_ALT, line=AMBER, line_w=2.0)
    for ax in (x1 + w1, x2 + w2):
        _arrow(s, ax + 25000, ya + ha // 2 - 110000, gap - 50000, 220000)
    _textbox(s, L, ya + ha + 40000, W, 280000, [P(
        "Did not finish: over 39 h of accumulated CPU; cancelled", 13, True, AMBER)])

    # After
    yb, hb = 2860000, 950000
    _pill(s, L, 2570000, 2400000, 250000, "AFTER · APPROXIMATE SVM", ACCENT)
    _card(s, x1, yb, w1, hb, [P("Data", 14, True, TEXT)])
    _card(s, x2, yb, w2, hb, [P("StandardScaler", 14, True, TEXT)])
    _card(s, x3, yb, w3, hb, [
        P("Nystroem map", 14, True, TEXT, 3),
        P("RBF, n_components", 12, False, MUTED)], fill=PANEL_ALT, line=ACCENT, line_w=2.0)
    _card(s, x4, yb, w4, hb, [
        P("", 12, runs=[("Classifier: ", ACCENT, True),
                        ("linear SVM (hinge loss) + Platt calibration", TEXT, False)], after=4),
        P("", 12, runs=[("Regressor: ", ACCENT, True),
                        ("linear ε-insensitive SVR", TEXT, False)])],
        fill=PANEL_ALT, line=ACCENT, line_w=2.0)
    for ax in (x1 + w1, x2 + w2, x3 + w3):
        _arrow(s, ax + 25000, yb + hb // 2 - 110000, gap - 50000, 220000)
    _textbox(s, L, yb + hb + 40000, W, 280000, [P(medido["slide"], 13, True, GREEN)])

    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, 4280000, W, 440000, fill=None, line=AMBER,
           line_w=2.0, paras=[P("Approximate SVM (approximate kernel), not an exact-kernel SVM",
                                14, True, AMBER)],
           margins=(150000, 30000, 150000, 30000))

    _notas(s, (
        "The original SVM used exact SVC and SVR with an RBF kernel, whose cost grows between n squared and n cubed; with about 154,000 rows the October 5 run accumulated more than a day and 15 hours of CPU "
        "(over 39 hours) without finishing and was cancelled. The alternative is an approximate SVM: the data are standardized, the RBF kernel is approximated with a Nystroem map of n_components components, and on that map "
        "a linear margin classifier with hinge loss is trained, calibrated with Platt scaling to obtain probabilities, together with a linear regressor with an ε-insensitive loss (linear SVR). "
        f"The cost becomes linear in the number of rows and it has been measured: {medido['notas']} "
        "It must be stated clearly: this is not an exact-kernel SVM, it is an approximate SVM, and it is documented as such in the preprocessing descriptor (svm_aproximado_nystroem).\n\n"
        "Sources: docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md §2 and §3.1; swmm_resilience/ml/bench/models/svm_family.py (docstring and build_classifier/build_regressor); "
        "docs/resultados/2026-10-06-handoff-svm.md §2 (more than 1 day and 15 h of CPU, cancelled, ~154,000 rows); "
        + medido["fuente"]
    ))
    return s


# ---------------------------------------------------------------------------
# Slide 6: fair comparison
# ---------------------------------------------------------------------------
def slide_comparacion_justa(prs, con_resultados: bool, datos_parte2_dir=None):
    n_trials = _n_trials(datos_parte2_dir)
    pie = FOOTER_TEXT + (
        "  ·  results on the following slides" if con_resultados
        else "  ·  results: runs in progress")
    s = nueva_slide(
        prs, "How the fair comparison is run",
        "Three families under the same conditions; only the model changes", pie)
    cw = 2750000
    xs = [L, L + cw + 150000, L + 2 * (cw + 150000)]
    familias = [
        ("XGBoost", "Gradient-boosted trees", "Always presented", ACCENT),
        ("MLP", "Neural network", "Always presented", ACCENT),
        ("Approximate SVM", "Nystroem + linear SVM", "Third candidate", AMBER),
    ]
    for x, (nombre, desc, etiqueta, color) in zip(xs, familias):
        _card(s, x, Y0 + 20000, cw, 930000, [
            P(nombre, 16, True, ACCENT, 2), P(desc, 12, False, MUTED, 2),
            P(etiqueta, 11, True, color)],
            fill=PANEL_ALT, line=color, line_w=2.0)
        _arrow(s, x + cw // 2 - 150000, Y0 + 990000, 300000, 230000, "down", LINE)

    filas = [
        ("Same data", "tabular_v4_21 · 154,080 rows (prep_id ca706e27654e0cf5)"),
        ("Same budget", f"{n_trials} and the same inner CV by shape"),
        ("Same protocol", "LeaveOneShapeOut: 40 folds, one held-out shape per fold"),
        ("Same metrics", "Volume RMSE · F1 · regressor NSE · confusion matrix"),
    ]
    py, rh = Y0 + 1260000, 400000
    for i, (etiqueta, valor) in enumerate(filas):
        fill = PANEL if i % 2 == 0 else PANEL_ALT
        _shape(s, MSO_SHAPE.RECTANGLE, L, py + i * rh, W, rh, fill=fill, line=None)
        _textbox(s, L + 120000, py + i * rh, 2250000, rh, [P(etiqueta, 14, True, ACCENT)],
                 anchor=MSO_ANCHOR.MIDDLE)
        _textbox(s, L + 2400000, py + i * rh, W - 2500000, rh, [P(valor, 13, False, TEXT)],
                 anchor=MSO_ANCHOR.MIDDLE)

    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, 4180000, W, 520000, fill=None, line=AMBER,
           line_w=2.0,
           paras=[P("Conditional exploratory estimate: hyperparameters are chosen with all shapes",
                    13, True, AMBER)],
           margins=(150000, 30000, 150000, 30000))

    _notas(s, (
        "To compare the three families fairly, all use the same data (tabular_v4_21 contract, 154,080 rows), the same tuning budget with Optuna "
        "(same number of trials and the same inner CV by shape), the same evaluation protocol, LeaveOneShapeOut, and the same metrics. "
        "XGBoost and MLP are always presented; the approximate SVM enters as a third candidate with whatever result it obtains. "
        "The important caveat: hyperparameters are chosen using all shapes, so the LeaveOneShapeOut figure is optimistic equally for the three families. "
        "It serves to compare them with each other, not as independent confirmation; that is why it is declared a conditional exploratory estimate. "
        "A nested tuning by shape is left as future work.\n\n"
        "Sources: docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md §1, §3.2 and §3.3 (author decisions of 2026-10-12); "
        "04-informe-consolidado.md header (prep_id ca706e27654e0cf5, 154,080 rows) and §5 point 1."
    ))
    return s


# ---------------------------------------------------------------------------
# Results slides (only when the part 2 tables exist)
# ---------------------------------------------------------------------------
FAMILIAS = [("xgboost", "XGBoost"), ("mlp", "MLP"), ("svm", "Approximate SVM")]
REFERENCIA = "baseline_forma_mas_cercana"
COLOR_FAMILIA = {"xgboost": ACCENT, "mlp": GREEN, "svm": AMBER}


def _datos_parte2(datos_dir) -> dict:
    """Read the part 2 tables (LeaveOneShapeOut) and index them by model."""
    datos_dir = Path(datos_dir)
    filas = [r for r in cargar_csv(datos_dir / "comparacion_protocolos.csv")
             if r["protocolo"] == "LeaveOneShapeOut"]
    if not filas:
        raise ValueError(f"{datos_dir} has no rows for the LeaveOneShapeOut protocol")
    por_modelo = {r["modelo"]: r for r in filas}
    for familia, _ in FAMILIAS:
        for variante in ("default", "tuned"):
            if f"{familia}_{variante}" not in por_modelo:
                raise ValueError(f"{familia}_{variante} is missing in comparacion_protocolos.csv")
    if REFERENCIA not in por_modelo:
        raise ValueError(f"{REFERENCIA} is missing in comparacion_protocolos.csv")
    matriz = {r["modelo"]: r for r in cargar_csv(datos_dir / "matriz_confusion.csv")
              if r["protocolo"] == "LeaveOneShapeOut"}
    banda = {r["modelo"]: r for r in cargar_csv(datos_dir / "banda_limitrofe.csv")
             if r["protocolo"] == "LeaveOneShapeOut"}
    resumen = {r["family"]: r for r in cargar_csv(datos_dir / "resumen_afinacion.csv")}
    ensayos: dict = {}
    for r in cargar_csv(datos_dir / "convergencia_afinacion.csv"):
        ensayos.setdefault(r["family"], []).append((int(r["trial"]), float(r["value"])))
    convergencia: dict = {}
    for familia, lista in ensayos.items():
        mejor, serie = float("inf"), []
        for _, valor in sorted(lista):  # running minimum per trial
            mejor = min(mejor, valor)
            serie.append(mejor)
        convergencia[familia] = serie
    return {"cp": por_modelo, "mc": matriz, "bl": banda, "resumen": resumen,
            "convergencia": convergencia}


def _slide_resultados(prs, d: dict, ruta: Path):
    cp = d["cp"]
    s = nueva_slide(
        prs, "XGBoost and MLP: results on unseen shapes",
        "LeaveOneShapeOut · XGBoost and MLP are the presented pair; approximate SVM, "
        "third candidate")

    filas = []
    for familia, nombre in FAMILIAS:
        for variante, sufijo in (("tuned", "tuned"), ("default", "default")):
            r = cp[f"{familia}_{variante}"]
            presentado = familia != "svm"
            color = ACCENT if presentado else TEXT
            corto = "SVM" if familia == "svm" else nombre
            filas.append([
                [(f"{corto} {sufijo}", color, presentado and variante == "tuned")],
                [(fmt(r["rmse_vol_agregado_m3"]), color, False)],
                [(fmt(r["f1_agregado"], 4), color, False)],
                [(fmt(r["auc_roc_agregado"], 4), color, False)],
                [(fmt(r["nse_regresor_oracle"], 3), color, False)],
            ])
    ref = cp[REFERENCIA]
    filas.append([[("No-learning baseline", AMBER, True)],
                  [(fmt(ref["rmse_vol_agregado_m3"]), AMBER, False)],
                  [(fmt(ref["f1_agregado"], 4), AMBER, False)],
                  [(fmt(ref["auc_roc_agregado"], 4), AMBER, False)],
                  [(fmt(ref["nse_regresor_oracle"], 3), AMBER, False)]])
    _tabla(s, L, Y0, [1900000, 800000, 750000, 750000, 650000],
           ["Model", "RMSE", "F1", "AUC", "NSE"], filas, row_h=370000, header_h=340000,
           size=12, name="Table results")
    _textbox(s, L, Y0 + 340000 + 370000 * len(filas) + 110000, 4850000, 420000, [
        P("Pooled volume RMSE in m³ (lower is better); regressor NSE on flooded nodes. "
          "Bars label the tuned value; default values are in the table", 10, False, MUTED)])

    cx, cw = L + 5000000, W - 5000000
    nombres = [n for _, n in FAMILIAS]
    for bloque, (rotulo, campo, decimales) in enumerate((
            ("Volume RMSE (m³): lower is better", "rmse_vol_agregado_m3", 2),
            ("Pooled F1: higher is better", "f1_agregado", 4))):
        y = Y0 + bloque * 1840000
        _textbox(s, cx, y, cw, 420000, [
            P(rotulo, 11, False, MUTED, 1),
            P(f"No-learning baseline: {fmt(ref[campo], decimales)}", 10, True, AMBER)])
        defecto = [round(float(cp[f"{f}_default"][campo]), decimales + 1) for f, _ in FAMILIAS]
        afinado = [round(float(cp[f"{f}_tuned"][campo]), decimales + 1) for f, _ in FAMILIAS]
        _chart_barras(
            s, cx, y + 400000, cw, 1380000, nombres,
            [("Default", defecto, LINE), ("Tuned", afinado, ACCENT)],
            etiquetas=[None, [fmt(v, decimales) for v in afinado]],
            colores_puntos=[[LINE, LINE, "3F5A78"], [ACCENT, ACCENT, "7A8FA6"]],
            horizontal=True, size=10, gap=40, maximo=max(defecto + afinado) * 1.45)

    x_t, x_d, mlp_t = cp["xgboost_tuned"], cp["xgboost_default"], cp["mlp_tuned"]
    mejora = 100 * (1 - float(x_t["rmse_vol_agregado_m3"]) / float(x_d["rmse_vol_agregado_m3"]))
    svm_t = cp["svm_tuned"]
    _notas(s, (
        "Results of the main protocol, LeaveOneShapeOut, with the same data, the same Optuna budget, and the same metrics for the three families. "
        f"Tuned XGBoost is the best: volume RMSE of {fmt(x_t['rmse_vol_agregado_m3'])} m³, {fmt(mejora, 0)}% lower than with its default parameters ({fmt(x_d['rmse_vol_agregado_m3'])} m³). "
        f"Tuned MLP comes second with {fmt(mlp_t['rmse_vol_agregado_m3'])} m³; both have an F1 close to {fmt(x_t['f1_agregado'], 2)} and an AUC close to {fmt(x_t['auc_roc_agregado'], 3)}. "
        f"The approximate SVM falls below even the no-learning baseline ({fmt(svm_t['rmse_vol_agregado_m3'])} against {fmt(ref['rmse_vol_agregado_m3'])} m³; F1 {fmt(svm_t['f1_agregado'], 3)} against {fmt(ref['f1_agregado'], 3)}). "
        "These are conditional exploratory estimates: the hyperparameters were chosen with all shapes, so they are optimistic equally for the three families. "
        "Note: the default XGBoost of this run does not match the one in the audit (106.21 m³ in datos/comparacion_protocolos.csv); they are different runs and are not compared with each other.\n\n"
        f"Source: {_rel(ruta)}/comparacion_protocolos.csv (LeaveOneShapeOut rows; columns rmse_vol_agregado_m3, f1_agregado, auc_roc_agregado, nse_regresor_oracle); "
        "docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md §3.2-§3.3."
    ))
    return s


def _slide_afinacion(prs, d: dict, ruta: Path):
    resumen, conv = d["resumen"], d["convergencia"]
    n_trials = max(len(v) for v in conv.values())
    s = nueva_slide(
        prs, f"Optuna tuning: {n_trials} trials per family",
        f"{n_trials} trials per family, same data and protocol; exploratory estimate "
        "(hyperparameters chosen with all shapes)")

    cd = CategoryChartData()
    cd.categories = [str(i + 1) for i in range(n_trials)]
    orden = [f for f, _ in FAMILIAS if f in conv]
    for familia in orden:
        cd.add_series(dict(FAMILIAS)[familia], [round(v, 2) for v in conv[familia]])
    cw = 5000000
    _textbox(s, L, Y0, cw, 260000, [P("Best inner RMSE up to that trial (m³)", 11, False, MUTED)])
    frame = s.shapes.add_chart(XL_CHART_TYPE.LINE, Emu(L), Emu(Y0 + 250000), Emu(cw),
                               Emu(3250000), cd)
    chart = frame.chart
    chart.has_title = False
    chart.font.size = Pt(11)
    chart.font.name = FONT
    chart.font.color.rgb = _rgb(TEXT)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(11)
    chart.legend.font.color.rgb = _rgb(TEXT)
    va = chart.value_axis
    va.minimum_scale = 0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = _rgb("2A4A6B")
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(10)
    va.tick_labels.font.color.rgb = _rgb(MUTED)
    va.tick_labels.number_format = "0"
    va.tick_labels.number_format_is_linked = False
    ca = chart.category_axis
    ca.format.line.color.rgb = _rgb(LINE)
    ca.tick_labels.font.size = Pt(10)
    ca.tick_labels.font.color.rgb = _rgb(MUTED)
    ca.has_major_gridlines = False
    for serie, familia in zip(chart.plots[0].series, orden):
        serie.smooth = False
        serie.format.line.color.rgb = _rgb(COLOR_FAMILIA[familia])
        serie.format.line.width = Pt(2.5)
        serie.marker.style = XL_MARKER_STYLE.NONE

    tx = L + cw + 200000
    tw = W - cw - 200000
    filas = []
    for familia in orden:
        r = resumen[familia]
        filas.append([
            [(dict(FAMILIAS)[familia].replace("Approximate SVM", "Approx. SVM"),
              COLOR_FAMILIA[familia], True)],
            [(fmt(r["best_value"], 1), TEXT, False)],
            [(fmt(float(r["duration_s"]) / 60, 1), TEXT, False)],
        ])
    _tabla(s, tx, Y0 + 250000, [1350000, 1150000, tw - 2500000],
           ["Family", "Best RMSE", "Min"], filas, row_h=480000, header_h=340000, size=12,
           name="Table tuning")
    _textbox(s, tx, Y0 + 250000 + 340000 + 480000 * len(filas) + 120000, tw, 900000, [
        P("Inner RMSE: GroupKFold(5) by shape inside the search; not the "
          "LeaveOneShapeOut figure.", 10, False, MUTED, 4),
        P("Min: duration of the whole study.", 10, False, MUTED)])

    detalle = "; ".join(
        f"{dict(FAMILIAS)[f]} ends at {fmt(resumen[f]['best_value'], 1)} m³ after "
        f"{fmt(float(resumen[f]['duration_s']) / 60, 1)} min" for f in orden)
    _notas(s, (
        f"Each family received the same search: {n_trials} Optuna trials on the same data and with the same inner CV by shape. "
        f"The curve shows the best inner RMSE reached up to each trial: {detalle}. "
        "The approximate SVM stays far from the other two throughout the search. "
        "It is a conditional exploratory estimate: hyperparameters are chosen with all shapes, and the inner value is not the LeaveOneShapeOut one.\n\n"
        f"Sources: {_rel(ruta)}/convergencia_afinacion.csv (family, trial, value; running minimum) and resumen_afinacion.csv (n_trials, best_value, duration_s, status)."
    ))
    return s


def _panel_matriz(s, x, y, ancho, titulo, matriz, metricas, banda):
    tn, fp, fn, tp = (int(matriz[k]) for k in ("tn", "fp", "fn", "tp"))
    _textbox(s, x, y, ancho, 300000, [P(titulo, 16, True, ACCENT)])
    rh, gap = 950000, 40000
    cell_w, cell_h = (ancho - rh - gap) // 2, 800000
    mx = x + rh
    for i, texto in enumerate(("no flood", "flood")):
        _textbox(s, mx + i * (cell_w + gap), y + 340000, cell_w, 380000,
                 [P("PREDICTED", 9, True, MUTED), P(texto, 12, True, TEXT)])
    celdas = [
        (tn, "TN", PANEL_ALT, LINE, 1.0, TEXT), (fp, "FP", PANEL, AMBER, 1.0, AMBER),
        (fn, "FN", PANEL, AMBER, 3.0, AMBER), (tp, "TP", HEAD, ACCENT, 1.5, TEXT),
    ]
    for idx, (valor, sigla, fill, line, line_w, color) in enumerate(celdas):
        fila, col = divmod(idx, 2)
        cy = y + 740000 + fila * (cell_h + gap)
        _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, mx + col * (cell_w + gap), cy, cell_w, cell_h,
               fill, line, line_w,
               paras=[P(fmt(valor, 0), 20, True, color, 1), P(sigla, 11, False, MUTED)],
               anchor=MSO_ANCHOR.TOP, margins=(110000, 70000, 80000, 30000))
    for fila, texto in enumerate(("no flood", "flood")):
        _textbox(s, x, y + 740000 + fila * (cell_h + gap), rh - 60000, cell_h,
                 [P("ACTUAL", 9, True, MUTED), P(texto, 12, True, TEXT)], anchor=MSO_ANCHOR.MIDDLE)
    ym = y + 740000 + 2 * cell_h + gap + 90000
    _textbox(s, x, ym, ancho, 280000, [P("", 12, runs=[
        ("Recall ", MUTED, False), (fmt(metricas["recall_agregado"], 4), TEXT, True),
        ("  ·  Precision ", MUTED, False), (fmt(metricas["precision_agregada"], 4), TEXT, True),
        ("  ·  F1 ", MUTED, False), (fmt(metricas["f1_agregado"], 4), ACCENT, True)])])
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, ym + 330000, ancho, 480000, fill=PANEL,
           line=AMBER, line_w=1.5, paras=[P(
               f"FN in the 1 m³ band: {int(banda['fn_en_banda_1_2m3'])} of {int(banda['fn_total'])}",
               12, True, TEXT)], margins=(110000, 30000, 110000, 30000))


def _slide_matrices(prs, d: dict, ruta: Path):
    s = nueva_slide(
        prs, "Confusion matrices: tuned XGBoost and MLP",
        "LeaveOneShapeOut · 154,080 node-scenarios · FN in the 1 m³ band: "
        "see banda_limitrofe.csv")
    pw = (W - 300000) // 2
    detalle = []
    for i, (modelo, titulo) in enumerate((("xgboost_tuned", "Tuned XGBoost"),
                                          ("mlp_tuned", "Tuned MLP"))):
        _panel_matriz(s, L + i * (pw + 300000), Y0 + 20000, pw, titulo, d["mc"][modelo],
                      d["cp"][modelo], d["bl"][modelo])
        m, b = d["mc"][modelo], d["bl"][modelo]
        detalle.append(
            f"{titulo}: TN {fmt(m['tn'], 0)}, FP {fmt(m['fp'], 0)}, FN {fmt(m['fn'], 0)}, "
            f"TP {fmt(m['tp'], 0)}; FN in the 1 m³ band: {b['fn_en_banda_1_2m3']} of "
            f"{b['fn_total']}; FN with actual volume of 100 m³ or more: {b['fn_con_vol_ge_100m3']}")
    _notas(s, (
        "Pooled matrices of the two families being presented, with tuned hyperparameters, holding out one whole hydrograph shape at a time. "
        "They classify almost the same; the difference lies in the false positives and in the few large false negatives. "
        + " | ".join(detalle) + ". "
        "Most false negatives are nodes with a reported volume of 1 m³, the minimum the label counts as flooding.\n\n"
        f"Sources: {_rel(ruta)}/matriz_confusion.csv (tn, fp, fn, tp), comparacion_protocolos.csv (recall_agregado, precision_agregada, f1_agregado) "
        "and banda_limitrofe.csv (fn_total, fn_en_banda_1_2m3, fn_con_vol_ge_100m3)."
    ))
    return s


# ---------------------------------------------------------------------------
# Figure slides (parity plots, error by shape, flood maps) from docs/figures_part2
# ---------------------------------------------------------------------------
MODELOS_FIG = [("xgboost", "XGBoost", ACCENT), ("mlp", "MLP", "F2A65A"),
               ("svm", "SVM", GREEN)]
_PARA_REGENERAR = ("Slide 5 (title picture whose shapes are not versioned) and slide 34 "
                   "(four parity pictures with a Spanish run identifier in the title)")


def _datos_figuras(figuras_dir) -> dict:
    """Numbers behind the figures: figure_numbers.json plus the pooled table of INDEX.md."""
    figuras_dir = Path(figuras_dir)
    nums = json.loads((figuras_dir / "figure_numbers.json").read_text(encoding="utf-8"))
    faltan = {}
    for linea in (figuras_dir / "INDEX.md").read_text(encoding="utf-8").splitlines():
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        if len(celdas) == 7 and re.match(r"(XGBoost|MLP|Approx\. SVM)", celdas[0]):
            clave = {"XGBoost": "xgboost", "MLP": "mlp", "Approx. SVM": "svm"}[
                re.match(r"(XGBoost|MLP|Approx\. SVM)", celdas[0]).group(1)]
            faltan[clave] = int(celdas[6].replace(",", ""))
    if set(faltan) != {"xgboost", "mlp", "svm"}:
        raise ValueError("The pooled summary table of INDEX.md was not found")
    formas = nums["shapes"]
    stats = {}
    for m, _, _ in MODELOS_FIG:
        errs = [abs(f[f"err_pct_{m}"]) for f in formas]
        stats[m] = {"mediana_abs": statistics.median(errs), "mayor_20": sum(e > 20 for e in errs)}
    return {"nums": nums, "missed": faltan, "formas": {f["shape_id"]: f for f in formas},
            "stats_formas": stats}


def _imagen_ajustada(slide, path, x, y, max_w, max_h, centrar=True):
    """Picture scaled to fit inside (max_w, max_h) keeping its aspect ratio."""
    from PIL import Image
    with Image.open(path) as im:
        ancho, alto = im.size
    escala = min(max_w / ancho, max_h / alto)
    w, h = int(ancho * escala), int(alto * escala)
    dx = (max_w - w) // 2 if centrar else 0
    return slide.shapes.add_picture(str(path), Emu(x + dx), Emu(y), Emu(w), Emu(h))


def _tarjeta_valores(s, x, y, w, h, rotulo, valores, fill=PANEL, line=LINE):
    """Card with a label and one value per model, each in the model colour."""
    runs = []
    for i, (txt, color) in enumerate(valores):
        if i:
            runs.append(("  ·  ", MUTED, False))
        runs.append((txt, color, True))
    return _card(s, x, y, w, h, [P(rotulo, 10, False, MUTED, 2), P("", 15, runs=runs)],
                 fill=fill, line=line)


def _slide_paridad_nodos(prs, fig, figuras_dir):
    p = fig["nums"]["pooled"]
    s = nueva_slide(
        prs, "Parity per node: predicted vs SWMM volume",
        "LeaveOneShapeOut · tuned models, out-of-fold · 154,080 node-scenario rows")
    _imagen_ajustada(s, figuras_dir / "parity_nodes_3models.png", L, Y0, W, 3080000)
    gap = 120000
    w3 = (W - 2 * gap) // 3
    y = Y0 + 3150000
    col = {m: c for m, _, c in MODELOS_FIG}
    orden = [m for m, _, _ in MODELOS_FIG]
    _tarjeta_valores(s, L, y, w3, 560000, "R² · XGBoost / MLP / SVM",
                     [(fmt(p[m]["r2"], 3), col[m]) for m in orden])
    _tarjeta_valores(s, L + w3 + gap, y, w3, 560000, "Pooled RMSE (m³)",
                     [(fmt(p[m]["rmse_m3"]), col[m]) for m in orden])
    _tarjeta_valores(s, L + 2 * (w3 + gap), y, w3, 560000, "SWMM-flooded rows missed",
                     [(fmt(fig["missed"][m], 0), col[m]) for m in orden], line=AMBER)
    r2 = [fmt(p[m]["r2"], 3) for m in orden]
    _notas(s, (
        "Each panel compares the predicted routed volume of every node-scenario row with the SWMM volume, for the three tuned models, always on shapes the model never saw. "
        f"XGBoost and MLP hug the diagonal over four orders of magnitude; the pooled R² is {r2[0]} for XGBoost, {r2[1]} for the MLP and {r2[2]} for the approximate SVM, and the RMSE is {fmt(p['xgboost']['rmse_m3'])}, {fmt(p['mlp']['rmse_m3'])} and {fmt(p['svm']['rmse_m3'])} m³. "
        f"The SVM is visibly wider around the diagonal and has two failure strips: it misses {fmt(fig['missed']['svm'], 0)} flooded rows (XGBoost {fmt(fig['missed']['xgboost'], 0)}) and predicts flooding on dry rows. "
        "RMSE is dominated by a few large-volume rows that the density plot hides, so quote it together with the scenario-level figure on the next slide. "
        "Caveat: the axes are log10(1 + volume), which makes small-volume errors look larger than they are in absolute terms.\n\n"
        "Sources: docs/figures_part2/parity_nodes_3models.png; docs/figures_part2/INDEX.md (pooled summary table, section 1); "
        "docs/figures_part2/figure_numbers.json (pooled)."
    ))
    return s


def _slide_paridad_escenarios(prs, fig, figuras_dir):
    n = fig["nums"]
    mediana, p90 = n["scenario_median_ape_pct"], n["scenario_p90_ape_pct"]
    s = nueva_slide(
        prs, "Parity per scenario: total flooded volume",
        f"One marker per scenario (shape × factor), {n['scenarios_total']} scenarios · "
        "log-log axes · out-of-fold")
    img = _imagen_ajustada(s, figuras_dir / "parity_scenario_totals_3models.png", L, Y0,
                           3650000, 3700000, centrar=False)
    px = L + img.width + 200000
    pw = L + W - px
    col = {m: c for m, _, c in MODELOS_FIG}
    orden = [m for m, _, _ in MODELOS_FIG]
    _tarjeta_valores(s, px, Y0, pw, 900000,
                     f"Median scenario error, % ({n['scenarios_swmm_ge_1m3']} scenarios ≥ 1 m³)"
                     " · XGBoost / MLP / SVM",
                     [(fmt(mediana[m], 1) + "%", col[m]) for m in orden])
    _card(s, px, Y0 + 1000000, pw, 1330000, [
        P("An honest split between metrics", 13, True, ACCENT, 3),
        P(f"MLP has the lower median scenario error ({fmt(mediana['mlp'], 1)}% vs "
          f"{fmt(mediana['xgboost'], 1)}%), while XGBoost has the lower pooled RMSE "
          f"({fmt(n['pooled']['xgboost']['rmse_m3'])} vs {fmt(n['pooled']['mlp']['rmse_m3'])} m³): "
          "they answer different questions", 12, False, TEXT)],
          fill=PANEL_ALT, line=ACCENT, line_w=2.0, anchor=MSO_ANCHOR.TOP)
    _card(s, px, Y0 + 2430000, pw, 1270000, [
        P("The SVM saturates at the top", 13, True, AMBER, 3),
        P("On the five largest scenarios (SWMM 142,000–183,000 m³) it predicts about "
          "67,000–75,000 m³, and it predicts volume in 102 of the 173 scenarios that are dry "
          "in SWMM (XGBoost 3, MLP 2)", 12, False, TEXT)],
          fill=PANEL, line=AMBER, line_w=1.5, anchor=MSO_ANCHOR.TOP)
    _notas(s, (
        "Here each marker is a whole scenario, and the distance to the dashed line is the relative error of its total flooded volume. "
        f"All three models follow the diagonal from about 10 to 180,000 m³; the median absolute percentage error is {fmt(mediana['xgboost'], 1)}% for XGBoost, {fmt(mediana['mlp'], 1)}% for the MLP and {fmt(mediana['svm'], 1)}% for the SVM (90th percentile {fmt(p90['xgboost'], 1)}% / {fmt(p90['mlp'], 1)}% / {fmt(p90['svm'], 1)}%). "
        "To be honest about it: at this level the MLP is slightly better than XGBoost on the median, even though XGBoost is clearly better on the pooled node RMSE; the two metrics answer different questions. "
        "The SVM saturates on the largest scenarios, all shape_quasi_uniform_block at factors 4 to 5, predicting about 67,000 to 75,000 m³ where SWMM gives 142,000 to 183,000 m³. "
        "Caveat: the median is taken over the 790 scenarios with SWMM total of at least 1 m³; values below 1 m³ are drawn at 1 m³.\n\n"
        "Sources: docs/figures_part2/parity_scenario_totals_3models.png; docs/figures_part2/figure_numbers.json (scenario_median_ape_pct, scenario_p90_ape_pct, scenarios_swmm_ge_1m3); "
        "docs/figures_part2/INDEX.md (section 2)."
    ))
    return s


def _slide_error_por_forma(prs, fig, figuras_dir):
    f, st = fig["formas"], fig["stats_formas"]
    q, cb = f["shape_quasi_uniform_block"], f["common_bogota"]
    ex = {"xgboost": st["xgboost"], "mlp": st["mlp"], "svm": st["svm"]}
    s = nueva_slide(
        prs, "Where the models fail: error by shape",
        "Total flood volume error per hydrograph shape (all factors pooled) · "
        "40 shapes · out-of-fold")
    img = _imagen_ajustada(s, figuras_dir / "error_by_shape.png", L, Y0, 3400000, 3700000,
                           centrar=False)
    px = L + img.width + 200000
    pw = L + W - px

    def pct(forma, sig=True):
        return " · ".join(f"{forma['err_pct_' + m]:+.1f}%" for m, _, _ in MODELOS_FIG)

    vol_q = fmt(q["vol_swmm"] / 1e6, 2)
    sse_total = sum(r["sse_xgboost"] for r in f.values())
    parte_sse = fmt(100 * q["sse_xgboost"] / sse_total, 0)
    _card(s, px, Y0, pw, 1380000, [
        P("A few shapes concentrate the failures", 13, True, ACCENT, 3),
        P(f"shape_quasi_uniform_block is under-predicted by all three "
          f"(XGBoost / MLP / SVM: {pct(q)}). It carries the largest SWMM volume "
          f"({vol_q} million m³) and holds {parte_sse}% of XGBoost's squared error",
          12, False, TEXT)],
          fill=PANEL_ALT, line=ACCENT, line_w=2.0, anchor=MSO_ANCHOR.TOP)
    _card(s, px, Y0 + 1480000, pw, 1100000, [
        P("common_bogota is over-predicted", 13, True, AMBER, 3),
        P(f"{pct(cb)}; a small-volume shape ({fmt(cb['vol_swmm'], 0)} m³ in SWMM), so its "
          "absolute impact is small", 12, False, TEXT)],
          fill=PANEL, line=AMBER, line_w=1.5, anchor=MSO_ANCHOR.TOP)
    _card(s, px, Y0 + 2680000, pw, 1020000, [
        P("Across shapes", 13, True, MUTED, 3),
        P("Median |error|: " + " · ".join(fmt(ex[m]["mediana_abs"], 1) + "%" for m, _, _ in MODELOS_FIG)
          + ". Shapes with |error| > 20%: " + " · ".join(str(ex[m]["mayor_20"]) for m, _, _ in MODELOS_FIG)
          + " (XGBoost / MLP / SVM)", 12, False, TEXT)],
          fill=PANEL, line=LINE, anchor=MSO_ANCHOR.TOP)
    _notas(s, (
        "This chart breaks the error down by hydrograph shape: bars to the right are over-prediction and to the left under-prediction, sorted by the absolute XGBoost error. "
        f"The failures are concentrated in a few shapes rather than spread evenly: shape_quasi_uniform_block is under-predicted by all three models ({pct(q)} for XGBoost, MLP and SVM) and carries the largest SWMM volume of all shapes, {vol_q} million m³; it also holds {parte_sse}% of the XGBoost squared error summed over shapes (sse_xgboost in figure_numbers.json), so it weighs heavily in the pooled RMSE. "
        f"common_bogota is over-predicted by all three ({pct(cb)}), but it is a small-volume shape. "
        f"The MLP has the smaller median shape error ({fmt(ex['mlp']['mediana_abs'], 1)}% vs {fmt(ex['xgboost']['mediana_abs'], 1)}% for XGBoost) but a larger worst case on common_bogota. "
        "Caveat: the error is pooled over all factors of a shape, so over- and under-prediction at different factors can cancel.\n\n"
        "Sources: docs/figures_part2/error_by_shape.png; docs/figures_part2/figure_numbers.json (shapes: err_pct_*, vol_swmm); docs/figures_part2/INDEX.md (section 3)."
    ))
    return s


_REGLAS_MAPA = {
    "max_error": ("Flood map: largest-error shape",
                  "Largest XGBoost error of the 40 shapes"),
    "median_error": ("Flood map: median-error shape",
                     "Median XGBoost error (position 21 of 40 shapes)"),
    "min_error": ("Flood map: smallest-error shape",
                  "Smallest XGBoost error (shapes with ≥ 20 flooded rows)"),
}
_DECIR_MAPA = {
    "max_error": (
        "The spatial pattern is right but the magnitude is systematically low for all three models. "
        "XGBoost finds the right set of 61 flooded nodes, with no false and no missed nodes, but underestimates the hot spots; "
        "the largest node, about 6,000 m³ in SWMM, is about 4,500 m³ in XGBoost. "
        "The SVM misclassifies 20 nodes."),
    "median_error": (
        "XGBoost and MLP reproduce the spatial pattern node by node: the same 60 flooded nodes and a node RMSE of about 20 m³. "
        "The SVM misses 10 flooded nodes, invents 5 and has a node RMSE about four times larger."),
    "min_error": (
        "The squared error is low here and the volumes are tiny, 283 m³ in total; in relative terms it is not an easy scenario: "
        "every model over-predicts and flags 20 to 23 extra nodes. The rule picks the lowest absolute squared error, which tends to favor low-volume shapes, so this is not a typical good case."),
}


def _slide_mapa(prs, fig, figuras_dir, regla):
    sel = next(x for x in fig["nums"]["selected_scenarios"] if x["rule"] == regla)
    st = fig["nums"]["scenario_stats"][regla]
    titulo, regla_txt = _REGLAS_MAPA[regla]
    sufijo = f"{regla}_{sel['shape']}_f{sel['factor']:.1f}"
    s = nueva_slide(
        prs, titulo,
        f"{regla_txt} · {sel['shape']} · factor {sel['factor']:.1f}")
    _imagen_ajustada(s, figuras_dir / f"flood_map_{sufijo}.png", L, Y0, W, 2400000)
    gap = 110000
    w4 = (W - 3 * gap) // 4
    y, h = Y0 + 2500000, 1180000
    _card(s, L, y, w4, h, [
        P("SWMM", 13, True, MUTED, 2), P(f"{fmt(st['swmm_total_m3'], 0)} m³", 15, True, TEXT, 1),
        P(f"{st['swmm_flooded_nodes']} flooded nodes", 12, False, MUTED)],
        fill=PANEL, line=LINE, anchor=MSO_ANCHOR.TOP)
    detalle = []
    for i, (m, nombre, color) in enumerate(MODELOS_FIG, start=1):
        d = st[m]
        _card(s, L + i * (w4 + gap), y, w4, h, [
            P(nombre, 13, True, color, 2),
            P(f"{fmt(d['total_m3'], 0)} m³ ({d['total_error_pct']:+.1f}%)", 12, True, TEXT, 1),
            P(f"+{d['false_positive_nodes']} false · −{d['missed_nodes']} missed", 11, False, MUTED, 1),
            P(f"Node RMSE {fmt(d['rmse_nodes_m3'], 1)} m³", 11, False, MUTED)],
            fill=PANEL, line=color, line_w=1.5, anchor=MSO_ANCHOR.TOP)
        detalle.append(f"{nombre} {fmt(d['total_m3'], 0)} m³ ({d['total_error_pct']:+.1f}%), +{d['false_positive_nodes']} false and −{d['missed_nodes']} missed nodes, node RMSE {fmt(d['rmse_nodes_m3'], 1)} m³")
    _notas(s, (
        f"This is one scenario, chosen by a documented rule and not by hand: {regla_txt[0].lower() + regla_txt[1:]}, shape {sel['shape']} at factor {sel['factor']:.1f}. "
        f"SWMM floods {st['swmm_flooded_nodes']} nodes with {fmt(st['swmm_total_m3'], 0)} m³ in total; "
        + "; ".join(detalle) + ". "
        + _DECIR_MAPA[regla] + " "
        "To read the map, compare the color and size of the same node across panels; an orange or green marker on a node that is grey in the SWMM panel is a false positive. "
        "The colors use a square-root scale shared by the four panels.\n\n"
        f"Sources: docs/figures_part2/flood_map_{sufijo}.png (the matching per-node bar chart is docs/figures_part2/node_profile_{sufijo}.png, not embedded); "
        "docs/figures_part2/figure_numbers.json (selected_scenarios, scenario_stats); docs/figures_part2/INDEX.md (section 4)."
    ))
    return s


def agregar_slides_figuras(prs, figuras_dir):
    """Parity (nodes and scenarios), error by shape and three flood-map slides."""
    figuras_dir = Path(figuras_dir)
    fig = _datos_figuras(figuras_dir)
    return [
        _slide_paridad_nodos(prs, fig, figuras_dir),
        _slide_paridad_escenarios(prs, fig, figuras_dir),
        _slide_error_por_forma(prs, fig, figuras_dir),
        _slide_mapa(prs, fig, figuras_dir, "max_error"),
        _slide_mapa(prs, fig, figuras_dir, "median_error"),
        _slide_mapa(prs, fig, figuras_dir, "min_error"),
    ]


# ---------------------------------------------------------------------------
# English pictures for the base deck (output deck only)
# ---------------------------------------------------------------------------
FIGURAS_EN_MAPA = {
    # the English Gantt has a wider label panel: drop the original crop and fit the box
    (2, "Imagen 7"): {"file": "slide02_gantt_schedule_en.png",
                      "box": (190000, 1150000, 8760000, 3980000)},
    (3, "Imagen 2"): "slide03_hydrograph_shapes_en.png",
    (4, "Imagen 4"): "slide03_hydrograph_shapes_en.png",
    (6, "Imagen 2"): "slide06_new_training_shapes_en.png",
    (11, "Imagen 5"): "slide11_validation_shapes_en.png",
    (15, "Imagen 4"): "slide15_flowchart_training_en.png",
    (15, "Imagen 12"): "slide15_flowchart_validation_en.png",
}


def replace_base_pictures(prs, mapping: dict, base_dir=None) -> int:
    """Swap the image of picture shapes, keeping geometry and crop.

    ``mapping`` = {(slide number, picture name): file}; relative files are
    resolved against ``base_dir``. A value may also be a dict
    {"file": ..., "box": (x, y, w, h)}: then the crop is removed and the
    picture is fitted (aspect kept, centered) inside that box. Returns the
    number of pictures replaced. Raises ValueError if a picture is not found."""
    n = 0
    for (numero, nombre), archivo in mapping.items():
        caja = None
        if isinstance(archivo, dict):
            caja, archivo = archivo.get("box"), archivo["file"]
        ruta = Path(archivo)
        if base_dir is not None and not ruta.is_absolute():
            ruta = Path(base_dir) / ruta
        if not ruta.exists():
            raise FileNotFoundError(ruta)
        slide = prs.slides[numero - 1]
        pics = [sh for sh in slide.shapes if sh.shape_type == 13 and sh.name == nombre]
        if not pics:
            raise ValueError(f"Picture {nombre!r} not found on slide {numero}")
        for pic in pics:
            blip = pic._element.blipFill.find(qn("a:blip"))
            antiguo = blip.get(qn("r:embed"))
            _, nuevo = slide.part.get_or_add_image_part(str(ruta))
            blip.set(qn("r:embed"), nuevo)
            if antiguo != nuevo:
                slide.part.drop_rel(antiguo)
            if caja is not None:
                from PIL import Image
                recorte = pic._element.blipFill.find(qn("a:srcRect"))
                if recorte is not None:
                    pic._element.blipFill.remove(recorte)
                with Image.open(ruta) as im:
                    ancho, alto = im.size
                escala = min(caja[2] / ancho, caja[3] / alto)
                w, h = int(ancho * escala), int(alto * escala)
                pic.left = Emu(caja[0] + (caja[2] - w) // 2)
                pic.top = Emu(caja[1] + (caja[3] - h) // 2)
                pic.width, pic.height = Emu(w), Emu(h)
            n += 1
    return n


def _slide_conclusiones(prs, d: dict, ruta: Path, fig: dict | None = None):
    cp, resumen = d["cp"], d["resumen"]
    xt, xd, mt = cp["xgboost_tuned"], cp["xgboost_default"], cp["mlp_tuned"]
    st, ref = cp["svm_tuned"], cp[REFERENCIA]
    mejora = 100 * (1 - float(xt["rmse_vol_agregado_m3"]) / float(xd["rmse_vol_agregado_m3"]))
    prov = {}
    ruta_prov = Path(ruta) / "tables_provenance.json"
    if ruta_prov.exists():
        prov = json.loads(ruta_prov.read_text(encoding="utf-8"))
    corridas = (f"{prov['n_runs']} of {prov['n_runs_full_grid']} runs"
                if "n_runs" in prov else "963 of 1000 runs")

    def fijos(familia):
        crudo = resumen.get(familia, {}).get("fixed_params") or "{}"
        return json.loads(crudo).get("classifier", {})

    mlp, xgb = fijos("mlp"), fijos("xgboost")
    alcance = (f"MLP: {mlp.get('epochs', 200)} epochs, batch {mlp.get('batch_size', 1024)} · "
               f"XGBoost: n_jobs={xgb.get('n_jobs', 1)}")

    # figure-based claims (only when the part 2 figures are available)
    r2 = mediana = None
    if fig is not None:
        r2 = fig["nums"]["pooled"]["xgboost"]["r2"]
        mediana = fig["nums"]["scenario_median_ape_pct"]

    s = nueva_slide(
        prs, "Conclusions and next steps",
        "Tuned XGBoost is the best on pooled RMSE; the approximate SVM falls below the baseline")
    tw = 4300000
    texto_xgb = (f"RMSE {fmt(xt['rmse_vol_agregado_m3'], 1)} m³ (−{fmt(mejora, 0)}% vs its "
                 f"default parameters)"
                 + (f" and node parity R² {fmt(r2, 2)}" if r2 is not None else "")
                 + f"; tuned MLP 2nd: {fmt(mt['rmse_vol_agregado_m3'], 1)} m³. ")
    if mediana is not None:
        texto_xgb += (f"But the MLP has the best median scenario error "
                      f"({fmt(mediana['mlp'], 1)}% vs {fmt(mediana['xgboost'], 1)}%). ")
    texto_xgb += (f"Both with F1 ≈ {fmt(xt['f1_agregado'], 2)} and "
                  f"AUC ≈ {fmt(xt['auc_roc_agregado'], 3)}")
    alto1 = 1300000 if mediana is not None else 1000000
    _card(s, L, Y0 + 20000, tw, alto1, [
        P("Tuned XGBoost is the best", 14, True, ACCENT, 2),
        P(texto_xgb, 12, False, TEXT)],
        fill=PANEL_ALT, line=ACCENT, line_w=2.0, anchor=MSO_ANCHOR.TOP)
    y2 = Y0 + 20000 + alto1 + 80000
    _card(s, L, y2, tw, 1250000, [
        P("SVM: third, below the baseline", 14, True, AMBER, 2),
        P(f"RMSE ≈ {fmt(st['rmse_vol_agregado_m3'], 0)} vs {fmt(ref['rmse_vol_agregado_m3'], 0)} m³; "
          f"F1 {fmt(st['f1_agregado'], 2)} vs {fmt(ref['f1_agregado'], 3)}. Plausible causes (not "
          "tested): approximate kernel, limited search space, Platt calibration neutralizing "
          "class_weight (seen on a toy problem only)", 12, False, TEXT)], fill=PANEL, line=AMBER, line_w=1.5,
          anchor=MSO_ANCHOR.TOP)
    y3 = y2 + 1250000 + 80000
    _card(s, L, y3, tw, Y0 + 3720000 - y3, [
        P("Honest scope", 14, True, MUTED, 2),
        P("Hyperparameters chosen with all shapes (optimistic equally) · one network · "
          f"{corridas} · {alcance}", 12, False, TEXT)], fill=PANEL, line=LINE,
          anchor=MSO_ANCHOR.TOP)

    px = L + tw + 250000
    pw = W - tw - 250000
    _textbox(s, px, Y0 + 20000, pw, 240000, [P("NEXT STEPS", 10, True, MUTED)])
    pasos = [
        "Decide the label: path A (recommended) or B",
        "Nested tuning by shape",
        "More trials and a wider search space for the SVM",
        "Confusion matrix for the extreme shapes",
        "Propagate to per-family maps",
        "Regenerate base-deck pictures with Spanish text (slides 5, 34)",
    ]
    pitch = 560000
    for i, paso in enumerate(pasos):
        py = Y0 + 280000 + i * pitch
        badge = _shape(s, MSO_SHAPE.OVAL, px, py + 50000, 380000, 380000, fill=ACCENT,
                       line=None, paras=[P(str(i + 1), 14, True, BG)], margins=(0, 0, 0, 0))
        badge.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        _card(s, px + 480000, py, pw - 480000, 480000, [P(paso, 12, False, TEXT)],
              fill=PANEL, line=LINE)

    escenario = (f" At scenario level the MLP has the lower median error ({fmt(mediana['mlp'], 1)}% vs {fmt(mediana['xgboost'], 1)}% for XGBoost), "
                 "so XGBoost does not win on every metric." if mediana is not None else "")
    _notas(s, (
        f"Main message: tuned XGBoost is the best on pooled volume RMSE, {fmt(xt['rmse_vol_agregado_m3'], 1)} m³, {fmt(mejora, 0)}% lower than with its default parameters"
        + (f", and on per-node parity (R² {fmt(r2, 2)})" if r2 is not None else "")
        + f"; tuned MLP comes second with {fmt(mt['rmse_vol_agregado_m3'], 1)} m³ and both have an F1 close to {fmt(xt['f1_agregado'], 2)} and an AUC close to {fmt(xt['auc_roc_agregado'], 3)}."
        + escenario +
        f" The approximate SVM falls below even the no-learning baseline: {fmt(st['rmse_vol_agregado_m3'], 0)} against {fmt(ref['rmse_vol_agregado_m3'], 0)} m³ and F1 {fmt(st['f1_agregado'], 2)} against {fmt(ref['f1_agregado'], 3)}; "
        "the plausible causes are the approximate kernel, a limited search space, and Platt calibration neutralizing class_weight, but none was tested on the real data: only the Platt/class_weight effect was measured, on a toy problem (mean predicted probability 0.429 with scale_pos_weight 1 vs 0.426 with 8), and it is declared as a known bias. "
        f"Scope: hyperparameters chosen with all shapes, one network, {corridas}; {alcance}. "
        "Next steps: decide the label (I recommend A), nested tuning by shape, more trials or a wider search space for the SVM, a confusion matrix for the extreme shapes, propagating to per-family maps, "
        "and regenerating the base-deck pictures that still contain Spanish text.\n\n"
        f"To regenerate: {_PARA_REGENERAR}. Their English versions were not produced because the shapes in slide 5 are not in any versioned data file and the slide 34 parity plots need a full out-of-fold re-evaluation of each family.\n\n"
        f"Sources: {_rel(ruta)}/comparacion_protocolos.csv, resumen_afinacion.csv (fixed_params), tables_provenance.json (n_runs, n_runs_full_grid); "
        "docs/figures_part2/figure_numbers.json and INDEX.md (R², median scenario error); docs/figures_en/INDEX.md (pictures to regenerate); "
        "docs/auditoria/correcciones/04-informe-consolidado.md §4 and §5; docs/auditoria/correcciones/05-bitacora-de-decisiones.md (P2-Task1 review: Platt vs class_weight on a toy problem; R22)."
    ))
    return s


def agregar_slide_resultados(prs, datos_dir, figuras_dir=None):
    """Add the part 2 results slides from the tables in ``datos_dir``:
    results (table + bars), tuning (convergence curves), confusion matrices,
    optionally the figure slides (parity, error by shape, flood maps) when
    ``figuras_dir`` exists, and conclusions/next steps. Returns the added
    slides. CSV column names are code identifiers (kept in Spanish)."""
    datos_dir = Path(datos_dir)
    d = _datos_parte2(datos_dir)
    nuevas = [
        _slide_resultados(prs, d, datos_dir),
        _slide_afinacion(prs, d, datos_dir),
        _slide_matrices(prs, d, datos_dir),
    ]
    fig = None
    if figuras_dir is not None and (Path(figuras_dir) / "figure_numbers.json").exists():
        nuevas += agregar_slides_figuras(prs, figuras_dir)
        fig = _datos_figuras(figuras_dir)
    nuevas.append(_slide_conclusiones(prs, d, datos_dir, fig))
    return nuevas


# ---------------------------------------------------------------------------
# English translation of the base deck (output deck only)
# ---------------------------------------------------------------------------
TRADUCCIONES = Path(__file__).resolve().parent / "deck_translations_en.json"

_ES_ACENTOS = re.compile("[áéíóúñ¿¡ÁÉÍÓÚÑ]")
_ES_PALABRAS = re.compile(
    r"\b(de|la|el|los|las|que|para|con|por|una|del|se|sin|más|como|entre|sobre|desde|"
    r"hacia|este|esta|esto|nuestro|nuestra|todas|todos|hidrograma|hidrogramas|"
    r"escenario|escenarios|corrida|corridas|filas|nodos|formas|familias|"
    r"resultados|siguiente|siguientes|presupuesto|computacional|medir|antes|"
    r"ejemplo|datos|ruta|aspecto|paso|costo|simulaciones|observaciones|cronograma|"
    r"tesis|pendiente|origen|valor|nota|familia|lineal|resultado|mejora|objetivo|"
    r"ganador|clasificador|regresor|fuente|c[oó]digo|referencias|nivel|acceso|"
    r"criterio|invertir|tiempo|b[uú]squeda|memoria|alcance|"
    r"recomendaci[oó]n|propia|cerrar|primero|reservar)\b",
    re.IGNORECASE)
_NOMBRES_PROPIOS = ("Garzón", "Brazão", "Paraíba", "Brasília", "Garzón")


def _parece_espanol(texto: str) -> bool:
    limpio = texto
    for nombre in _NOMBRES_PROPIOS:
        limpio = limpio.replace(nombre, "")
    if re.match(r"^\s*(https?://|\S+\.(png|csv|json)$)", limpio):
        return False
    return bool(_ES_ACENTOS.search(limpio) or _ES_PALABRAS.search(limpio))


def _cargar_traducciones(ruta=TRADUCCIONES) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _aplicar_a_parrafo(para, ingles: str) -> None:
    """English into the first run (keeps its formatting); the others are emptied."""
    runs = para.runs
    if not runs:
        return
    runs[0].text = ingles
    for extra in runs[1:]:
        extra.text = ""
    for run in runs:
        rpr = run._r.find(qn("a:rPr"))
        if rpr is not None and rpr.get("lang", "").startswith("es"):
            rpr.set("lang", "en-US")
            if rpr.get("err"):
                del rpr.attrib["err"]


def _marcos_de_texto(shapes):
    for sh in shapes:
        if sh.shape_type == 6:  # group
            yield from _marcos_de_texto(sh.shapes)
            continue
        if sh.has_text_frame:
            yield sh.text_frame
        if getattr(sh, "has_table", False) and sh.has_table:
            for row in sh.table.rows:
                for cell in row.cells:
                    yield cell.text_frame


def translate_base(prs, translations: dict | None = None, n_base: int | None = None) -> dict:
    """Translate to English every paragraph (slide text frames, table cells,
    speaker notes) and alt text of the first ``n_base`` slides. Paragraphs
    that look Spanish and have no entry are collected and reported: raises
    ValueError listing them. Numbers/names/files are not touched except by the
    exact translation entries. Returns counters."""
    traducciones = _cargar_traducciones() if translations is None else translations
    n_base = len(prs.slides) if n_base is None else n_base
    contadores = {"frames": 0, "notes": 0, "alt": 0}
    faltan: list[str] = []

    def _procesa(marco, tipo, donde):
        for para in marco.paragraphs:
            texto = "".join(r.text for r in para.runs)
            if not texto.strip():
                continue
            if texto in traducciones:
                if traducciones[texto] != texto:
                    _aplicar_a_parrafo(para, traducciones[texto])
                    contadores[tipo] += 1
            elif _parece_espanol(texto):
                faltan.append(f"{donde}: {texto[:90]}")

    for idx, slide in enumerate(list(prs.slides)[:n_base], start=1):
        for marco in _marcos_de_texto(slide.shapes):
            _procesa(marco, "frames", f"slide {idx}")
        if slide.has_notes_slide:
            _procesa(slide.notes_slide.notes_text_frame, "notes", f"slide {idx} notes")
        for el in slide._element.iter():
            descr = el.get("descr") if hasattr(el, "get") else None
            if descr and descr in traducciones and traducciones[descr] != descr:
                el.set("descr", traducciones[descr])
                contadores["alt"] += 1
    if faltan:
        raise ValueError("Spanish text without a translation entry:\n" + "\n".join(faltan))
    return contadores


# ---------------------------------------------------------------------------
def construir(base: Path = BASE, salida: Path = OUTPUT, datos_dir: Path = DATOS_DIR,
              datos_parte2_dir: Path = DATOS_PARTE2_DIR, traducir: bool = True,
              figuras_dir: Path | None = FIGURAS_DIR,
              figuras_en_dir: Path | None = FIGURAS_EN_DIR):
    base, salida = Path(base), Path(salida)
    if not base.exists():
        raise SystemExit(f"{base} does not exist")
    if salida.resolve() == base.resolve():
        raise SystemExit("The output cannot be the base presentation")
    prs = Presentation(str(base))
    n_base = len(prs.slides)
    if traducir:
        contadores = translate_base(prs, n_base=n_base)
        print(f"Base deck translated to English: {contadores}")
    if figuras_en_dir is not None and Path(figuras_en_dir).is_dir():
        mapa = {k: v for k, v in FIGURAS_EN_MAPA.items() if k[0] <= n_base}
        reemplazadas = replace_base_pictures(prs, mapa, figuras_en_dir)
        print(f"English pictures swapped into the base deck: {reemplazadas}")
    con_resultados = (Path(datos_parte2_dir) / "comparacion_protocolos.csv").exists()

    slide_auditoria(prs, datos_dir)
    slide_protocolos(prs, datos_dir)
    slide_matriz(prs, datos_dir)
    slide_etiqueta(prs)
    slide_svm(prs, datos_parte2_dir if con_resultados else None)
    slide_comparacion_justa(prs, con_resultados, datos_parte2_dir if con_resultados else None)
    if con_resultados:
        agregar_slide_resultados(prs, datos_parte2_dir, figuras_dir)

    prs.save(str(salida))
    return prs


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legacy-output", action="store_true",
                    help="write the old file name (Presentacion_MLSMFP_13-10_...) instead "
                         "of the 20-10 one; fails if that file is open in PowerPoint")
    args = ap.parse_args(argv)
    destino = OUTPUT_LEGACY if args.legacy_output else OUTPUT
    prs = construir(salida=destino)
    print(f"Saved: {destino} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
