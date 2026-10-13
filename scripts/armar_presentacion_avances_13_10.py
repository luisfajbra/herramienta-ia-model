"""Arma la presentacion del 13 de octubre de 2026 (avances: auditoria, SVM
aproximado y comparacion justa) a partir de la presentacion del 5 de octubre.

La presentacion base nunca se modifica: el script la abre, le agrega 6
diapositivas nuevas (mas 2 de resultados si ya existen las tablas de la
parte 2) y guarda el resultado en un archivo distinto. Es idempotente:
siempre reconstruye la salida desde la base.

Las cifras de las diapositivas de la auditoria salen de los CSV de
``docs/auditoria/correcciones/datos`` (se leen al construir); el resto es
texto de ``docs/auditoria/correcciones/0[1-4]-*.md`` y de la spec
``docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md``.

Uso:
    venv/Scripts/python.exe scripts/armar_presentacion_avances_13_10.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import (  # noqa: E402
    XL_CHART_TYPE,
    XL_LABEL_POSITION,
    XL_LEGEND_POSITION,
    XL_TICK_MARK,
)
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Pt  # noqa: E402

BASE = REPO_ROOT / "Presentacion_MLSMFP_05-10_resultados_fase_c2.pptx"
OUTPUT = REPO_ROOT / "Presentacion_MLSMFP_13-10_avances_auditoria_svm.pptx"
DATOS_DIR = REPO_ROOT / "docs" / "auditoria" / "correcciones" / "datos"
DATOS_PARTE2_DIR = REPO_ROOT / "docs" / "auditoria" / "correcciones" / "datos_parte2"

FOOTER_TEXT = (
    "MLSMFP  |  Avances: auditoría, SVM aproximado y comparación justa  |  13 octubre 2026"
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
    """Formato espanol: coma decimal y punto de miles (106526 -> 106.526)."""
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def cargar_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fila(rows: list[dict], modelo: str, protocolo: str) -> dict:
    for row in rows:
        if row["modelo"] == modelo and row["protocolo"] == protocolo:
            return row
    raise KeyError(f"No hay fila para {modelo} / {protocolo}")


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
            if etiquetas:
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
# Diapositiva 1: auditoria
# ---------------------------------------------------------------------------
def slide_auditoria(prs, datos_dir: Path = DATOS_DIR):
    cp = cargar_csv(Path(datos_dir) / "comparacion_protocolos.csv")
    xgb_loso = _fila(cp, "xgboost", "LOSO")
    ref_loso = _fila(cp, "baseline_interpolacion_factor", "LOSO")

    s = nueva_slide(
        prs, "Auditoría: qué se encontró y qué se corrigió",
        "Siete hallazgos de la auditoría del 10 de octubre · XGBoost por defecto, 154.080 filas",
    )

    def cod(codigo, resto):
        return [(codigo + "  ", ACCENT, True), (resto, TEXT, False)]

    estado = {"Corregido": GREEN, "Parcial": AMBER, "Abierto": AMBER, "Pendiente": AMBER}
    filas_raw = [
        (cod("M-01", "Sin referencias"),
         "Dos referencias sin aprendizaje en el banco",
         f"LOSO: referencia {fmt(ref_loso['rmse_vol_agregado_m3'])} m³ vs XGBoost "
         f"{fmt(xgb_loso['rmse_vol_agregado_m3'])} m³", "Corregido"),
        (cod("M-02", "Volumen de la forma"),
         "Medición por forma; el modelo no cambió",
         "Una forma = 71,6 % del error cuadrático", "Abierto"),
        (cod("M-03", "F1 promediado"),
         "F1 agregado sobre la matriz de confusión",
         f"LOSO: F1 {fmt(xgb_loso['f1_media_por_fold_con_ceros'], 4)} → "
         f"{fmt(xgb_loso['f1_agregado'], 4)}", "Corregido"),
        (cod("V-01", "Figuras en muestra"),
         "Pie «en muestra» en las figuras nuevas",
         "6 pruebas; figuras antiguas sin regenerar", "Parcial"),
        (cod("V-07", "Script sobrescribía el .inp"),
         "Sin efectos al importar; exige --force",
         "No sobrescribe sin --force (pruebas)", "Corregido"),
        (cod("M-07 / V-04", "Modelo ≠ promotion.json"),
         "predict_network compara los SHA-256",
         "Se detiene si no coinciden (15 pruebas)", "Parcial"),
        (cod("D-04", "Resolución de la etiqueta"),
         "Spike: umbral efectivo ≈ 0,5 m³",
         "Decisión A/B pendiente; recomiendo A", "Pendiente"),
    ]
    filas = [[a, b, c, [(d, estado[d], True)]] for a, b, c, d in filas_raw]
    _tabla(s, L, Y0, [2250000, 2400000, 2650000, 1250000],
           ["Hallazgo", "Corrección", "Verificación", "Estado"], filas,
           row_h=480000, header_h=340000, size=12)

    _notas(s, (
        "La auditoría interna del protocolo encontró siete problemas que afectaban cómo se leen los resultados; "
        "esta tabla resume cada hallazgo, su corrección y cómo se verificó. "
        "El más importante es M-01: en LOSO una referencia que solo interpola entre factores vecinos obtiene un RMSE de "
        f"{fmt(ref_loso['rmse_vol_agregado_m3'])} m³ contra {fmt(xgb_loso['rmse_vol_agregado_m3'])} m³ de XGBoost, así que ese protocolo mide interpolación y no generalización. "
        f"M-03 es de agregación: promediar el F1 por pliegue con ceros daba {fmt(xgb_loso['f1_media_por_fold_con_ceros'], 4)} y el F1 agregado es {fmt(xgb_loso['f1_agregado'], 4)}. "
        "Dos estados no son de cierre: M-02 sigue abierto (una sola forma concentra el 71,6 % del error cuadrático) y D-04 espera mi decisión entre las rutas A y B. "
        "V-01 y M-07/V-04 están corregidos solo en las figuras nuevas y en predict_network; otros cargadores siguen sin verificación.\n\n"
        "Fuentes: docs/auditoria/correcciones/04-informe-consolidado.md (§1 estado por hallazgo, §2 antes y después); "
        "datos/comparacion_protocolos.csv (rmse_vol_agregado_m3, f1_agregado, f1_media_por_fold_con_ceros, XGBoost y "
        "baseline_interpolacion_factor en LOSO); datos/por_forma.csv (71,6 %: rmse_vol^2 x n_rows de shape_quasi_uniform_block sobre el total, XGBoost, LeaveOneShapeOut); "
        "01-evaluacion-honesta.md §1-§3; 02-trazabilidad-y-seguridad.md §1-§3 (6 pruebas de V-01, 15 de verify_promotion); 03-resolucion-de-la-etiqueta.md §1."
    ))
    return s


# ---------------------------------------------------------------------------
# Diapositiva 2: que mide cada protocolo
# ---------------------------------------------------------------------------
def slide_protocolos(prs, datos_dir: Path = DATOS_DIR):
    cp = cargar_csv(Path(datos_dir) / "comparacion_protocolos.csv")
    ref_loso = _fila(cp, "baseline_interpolacion_factor", "LOSO")
    xgb_loso = _fila(cp, "xgboost", "LOSO")
    ref_forma = _fila(cp, "baseline_forma_mas_cercana", "LeaveOneShapeOut")
    xgb_forma = _fila(cp, "xgboost", "LeaveOneShapeOut")

    s = nueva_slide(
        prs, "Qué mide cada protocolo (y por qué importa)",
        "RMSE de volumen agregado, XGBoost por defecto frente a una referencia que no aprende",
    )
    card_w = 3950000
    _card(s, L, Y0 + 20000, card_w, 1330000, [
        P("LOSO y GroupKFold5", 14, True, TEXT, 3),
        P("Se retienen factores de intensidad", 12, False, MUTED, 5),
        P("Interpolación entre factores", 14, True, AMBER, 3),
        P("Todas las formas ya se vieron", 12, False, MUTED),
    ])
    _card(s, L, Y0 + 1480000, card_w, 1330000, [
        P("LeaveOneShapeOut", 14, True, TEXT, 3),
        P("Se retiene una forma de hidrograma completa", 12, False, MUTED, 5),
        P("Formas de hidrograma nuevas", 14, True, ACCENT, 3),
        P("40 pliegues: la prueba principal", 12, False, MUTED),
    ], fill=PANEL_ALT, line=ACCENT, line_w=2.25)

    cx = L + 4150000
    cw = W - 4150000
    _textbox(s, cx, Y0, cw, 260000,
             [P("RMSE agregado del volumen (m³): menor es mejor", 11, False, MUTED)])
    ref_loso_v = round(float(ref_loso["rmse_vol_agregado_m3"]), 2)
    xgb_loso_v = round(float(xgb_loso["rmse_vol_agregado_m3"]), 2)
    ref_forma_v = round(float(ref_forma["rmse_vol_agregado_m3"]), 2)
    xgb_forma_v = round(float(xgb_forma["rmse_vol_agregado_m3"]), 2)
    _chart_barras(
        s, cx, Y0 + 250000, cw, 2560000,
        ["LOSO", "LeaveOneShapeOut"],
        [("Referencia sin aprendizaje", [ref_loso_v, ref_forma_v], AMBER),
         ("XGBoost", [xgb_loso_v, xgb_forma_v], ACCENT)],
        etiquetas=[[fmt(ref_loso_v), fmt(ref_forma_v)], [fmt(xgb_loso_v), fmt(xgb_forma_v)]],
        size=12, gap=55,
    )
    _textbox(s, cx, Y0 + 2800000, cw, 200000, [P(
        "Ref.: interpolación (izq.), forma más cercana (der.)",
        10, False, MUTED)])

    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, 4080000, W, 650000, fill=HEAD, line=None,
           paras=[P("LeaveOneShapeOut es la cifra principal", 18, True, TEXT, 2),
                  P("En LOSO la referencia que no aprende gana: ese protocolo solo mide interpolación",
                    12, False, MUTED)],
           margins=(150000, 40000, 150000, 40000))

    _notas(s, (
        "Cada protocolo retiene un grupo distinto. En LOSO y GroupKFold5 se retienen factores de intensidad y todas las formas siguen en el entrenamiento, "
        "así que se mide interpolación entre factores; en LeaveOneShapeOut se retiene una forma completa y se mide generalización a formas nuevas. "
        f"La prueba de que importa: en LOSO la referencia que no aprende tiene RMSE de {fmt(ref_loso_v)} m³ y XGBoost {fmt(xgb_loso_v)} m³, es decir, ganar en LOSO no demuestra nada. "
        f"En LeaveOneShapeOut el orden se invierte: la referencia de forma más cercana tiene {fmt(ref_forma_v)} m³ y XGBoost {fmt(xgb_forma_v)} m³. "
        "Por eso LeaveOneShapeOut es la cifra principal de toda la comparación. Con la malla de 25 factores, GroupKFold5 no es más exigente que LOSO: ningún pliegue retiene dos factores contiguos.\n\n"
        "Fuentes: docs/auditoria/correcciones/datos/comparacion_protocolos.csv, columna rmse_vol_agregado_m3 "
        "(baseline_interpolacion_factor/LOSO, xgboost/LOSO, baseline_forma_mas_cercana/LeaveOneShapeOut, xgboost/LeaveOneShapeOut); "
        "01-evaluacion-honesta.md §0 (qué mide cada protocolo) y §1; 04-informe-consolidado.md §3 y §11."
    ))
    return s


# ---------------------------------------------------------------------------
# Diapositiva 3: matriz de confusion
# ---------------------------------------------------------------------------
def slide_matriz(prs, datos_dir: Path = DATOS_DIR):
    datos_dir = Path(datos_dir)
    mc = _fila(cargar_csv(datos_dir / "matriz_confusion.csv"), "xgboost", "LeaveOneShapeOut")
    cp = _fila(cargar_csv(datos_dir / "comparacion_protocolos.csv"), "xgboost", "LeaveOneShapeOut")
    bl = _fila(cargar_csv(datos_dir / "banda_limitrofe.csv"), "xgboost", "LeaveOneShapeOut")
    tn, fp, fn, tp = (int(mc[k]) for k in ("tn", "fp", "fn", "tp"))
    fn_banda, fn_total = int(bl["fn_en_banda_1_2m3"]), int(bl["fn_total"])

    s = nueva_slide(
        prs, "Matriz de confusión de XGBoost con formas no vistas",
        f"LeaveOneShapeOut · {fmt(tn + fp + fn + tp, 0)} nodos-escenario · etiqueta: volumen ≥ 1 m³",
    )
    rh_w, cell_w, cell_h, gap = 1250000, 1900000, 980000, 50000
    mx = L + rh_w + 50000
    my1, my2 = 1500000, 1500000 + cell_h + gap
    for i, texto in enumerate(("no inunda", "inunda")):
        _textbox(s, mx + i * (cell_w + gap), Y0 + 20000, cell_w, 400000,
                 [P("PREDICHO", 10, True, MUTED), P(texto, 14, True, TEXT)])
    for texto, y in (("no inunda", my1), ("inunda", my2)):
        _textbox(s, L, y, rh_w, cell_h, [P("REAL", 10, True, MUTED), P(texto, 14, True, TEXT)],
                 anchor=MSO_ANCHOR.MIDDLE)

    def celda(x, y, valor, etiqueta, fill, line, line_w, color_num):
        _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, cell_w, cell_h, fill, line, line_w,
               paras=[P(fmt(valor, 0), 24, True, color_num, 2), P(etiqueta, 12, False, MUTED)],
               anchor=MSO_ANCHOR.TOP, margins=(130000, 90000, 100000, 40000))

    celda(mx, my1, tn, "TN · verdaderos negativos", PANEL_ALT, LINE, 1.0, TEXT)
    celda(mx + cell_w + gap, my1, fp, "FP · falsos positivos", PANEL, AMBER, 1.0, AMBER)
    celda(mx, my2, fn, "FN · falsos negativos", PANEL, AMBER, 3.0, AMBER)
    celda(mx + cell_w + gap, my2, tp, "TP · verdaderos positivos", HEAD, ACCENT, 1.5, TEXT)

    tx = mx + 2 * cell_w + gap + 250000
    tw = L + W - tx
    tile_h = 780000
    metricas = [("Recall", fmt(cp["recall_agregado"], 4), ACCENT),
                ("Precisión", fmt(cp["precision_agregada"], 4), ACCENT),
                ("F1 agregado", fmt(cp["f1_agregado"], 4), TEXT)]
    for i, (nombre, valor, color) in enumerate(metricas):
        _card(s, tx, Y0 + 20000 + i * (tile_h + 60000), tw, tile_h,
              [P(nombre, 12, False, MUTED, 1), P(valor, 22, True, color)],
              fill=PANEL, line=LINE)

    cy, ch = 3680000, 950000
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, cy, W, ch, fill=PANEL, line=AMBER, line_w=2.0)
    _textbox(s, L + 150000, cy, 2900000, ch,
             [P(f"{fn_banda} de {fn_total}", 30, True, AMBER)], anchor=MSO_ANCHOR.MIDDLE)
    _textbox(s, L + 3150000, cy, W - 3300000, ch, [
        P("falsos negativos tienen volumen reportado de 1 m³", 14, True, TEXT, 3),
        P("el mínimo que cuenta como inundación; ninguno supera 100 m³", 12, False, MUTED),
    ], anchor=MSO_ANCHOR.MIDDLE)

    _notas(s, (
        "Esta es la matriz agregada de XGBoost con hiperparámetros por defecto cuando se retiene una forma de hidrograma completa, sobre "
        f"{fmt(tn + fp + fn + tp, 0)} casos nodo-escenario. Hay {fmt(tp, 0)} verdaderos positivos, {fmt(tn, 0)} verdaderos negativos, {fmt(fp, 0)} falsos positivos y {fmt(fn, 0)} falsos negativos; "
        f"de ahí salen recall {fmt(cp['recall_agregado'], 4)}, precisión {fmt(cp['precision_agregada'], 4)} y F1 {fmt(cp['f1_agregado'], 4)}. "
        f"Lo que quiero que se lleven: {fn_banda} de los {fn_total} falsos negativos son nodos cuyo volumen reportado es de 1 m³, el mínimo que la etiqueta considera inundación; "
        "ninguno supera los 100 m³. Es decir, los errores de clasificación se concentran en el borde de la etiqueta y no en inundaciones grandes. "
        "Si se excluye esa banda el F1 sube de 0,9817 a 0,9847, pero eso se reporta solo como sensibilidad, con exclusión asimétrica.\n\n"
        "Fuentes: docs/auditoria/correcciones/datos/matriz_confusion.csv (xgboost, LeaveOneShapeOut: tn, fp, fn, tp); "
        "datos/comparacion_protocolos.csv (recall_agregado, precision_agregada, f1_agregado); "
        "datos/banda_limitrofe.csv (fn_total, fn_en_banda_1_2m3, fn_con_vol_ge_100m3); 04-informe-consolidado.md §3 y §4; 03-resolucion-de-la-etiqueta.md §1 punto 6 (F1 0,9847 sin la banda)."
    ))
    return s


# ---------------------------------------------------------------------------
# Diapositiva 4: etiqueta y resolucion
# ---------------------------------------------------------------------------
def slide_etiqueta(prs):
    s = nueva_slide(
        prs, "La etiqueta de inundación y su resolución",
        "De SWMM al .rpt y a la etiqueta: el reporte imprime el volumen con resolución de 1 m³",
    )
    bw, gap_w = 2500000, 524820
    y, h = Y0 + 20000, 1000000
    cajas = [
        ("1 · SWMM", "Volumen de desborde continuo (coma flotante)"),
        ("2 · Reporte .rpt", "10⁶ L con 3 decimales: resolución de 1 m³"),
        ("3 · Etiqueta", "inunda = 1 si V_rpt ≥ 1 m³"),
    ]
    for i, (titulo, detalle) in enumerate(cajas):
        x = L + i * (bw + gap_w)
        _card(s, x, y, bw, h, [P(titulo, 14, True, ACCENT, 4), P(detalle, 13, False, TEXT)],
              fill=PANEL_ALT if i == 1 else PANEL, line=AMBER if i == 1 else LINE,
              line_w=2.0 if i == 1 else 1.0, anchor=MSO_ANCHOR.TOP)
        if i < 2:
            _arrow(s, x + bw + 70000, y + h // 2 - 150000, gap_w - 140000, 300000)

    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, 2200000, W, 800000, fill=HEAD, line=None,
           paras=[P("etiqueta ≈ volumen continuo ≥ 0,5 m³", 18, True, TEXT, 3),
                  P("acotado a 0,4988–0,5002 m³ en 15 escenarios de control (2.400 casos nodo-escenario)",
                    12, False, MUTED)],
           margins=(150000, 40000, 150000, 40000))

    _textbox(s, L, 3110000, W, 200000,
             [P("DECISIÓN PENDIENTE (DEL AUTOR)", 10, True, MUTED)])
    cw = (W - 150000) // 2
    cy, ch = 3450000, 1150000
    _card(s, L, cy, cw, ch, [
        P("A · Declarar el supuesto", 14, True, ACCENT, 4),
        P("Se mantiene el dato y se reporta la matriz con la banda de 1 m³ separada", 12, False, TEXT)],
        fill=PANEL_ALT, line=AMBER, line_w=2.25)
    _pill(s, L + cw - 2150000, cy - 110000, 2050000, 240000, "RECOMENDACIÓN: A", AMBER)
    _card(s, L + cw + 150000, cy, cw, ch, [
        P("B · Re-simular las 1000 corridas", 14, True, ACCENT, 4),
        P("≈ 17–20 min de SWMM, pero se rehacen dataset, BD, banco, Optuna y todas las cifras",
          12, False, TEXT)])

    _notas(s, (
        "La etiqueta de inundación no sale directamente del volumen continuo de SWMM: el reporte .rpt lo imprime en 10⁶ L con tres decimales, es decir con resolución de 1 m³, "
        "y la etiqueta es volumen reportado mayor o igual a 1 m³. En las re-simulaciones de control eso equivale a un umbral de volumen continuo de aproximadamente 0,5 m³, "
        "acotado a entre 0,4988 y 0,5002 m³ en 15 escenarios con registro por nodo, sin discrepancias en 2.400 casos nodo-escenario; es una cuantización determinista, no ruido. "
        "Hay dos rutas: A, mantener resolución y umbral y declararlos como supuesto reportando la matriz con la banda separada, o B, re-simular las 1000 corridas, que cuesta unos 17 a 20 minutos de SWMM "
        "pero obliga a rehacer dataset, base de datos, prep_id, banco, Optuna y todas las cifras. La recomendación es la ruta A; la decisión es mía y está pendiente.\n\n"
        "Fuentes: docs/auditoria/correcciones/03-resolucion-de-la-etiqueta.md §1 (puntos 1-8) y §2.1 (cadena de resolución); "
        "04-informe-consolidado.md §4. Cota 0,4988-0,5002 m³ y 2.400 casos: 03 §1 punto 2; 17-20 min: 03 §1 punto 5. "
        "La alerta de que la decisión sigue abierta está también en 04 §11 (Pendiente)."
    ))
    return s


# ---------------------------------------------------------------------------
# Diapositiva 5: arquitectura del SVM
# ---------------------------------------------------------------------------
def slide_svm(prs):
    s = nueva_slide(
        prs, "Cambio de arquitectura del SVM",
        "El kernel RBF exacto no escala a 154.080 filas; se aproxima con un mapeo de Nystroem",
    )
    gap = 190000
    x1, w1 = L, 1000000
    x2, w2 = x1 + w1 + gap, 1900000
    x3, w3 = x2 + w2 + gap, 2250000
    x4, w4 = x3 + w3 + gap, L + W - (x3 + w3 + gap)

    # Antes
    ya, ha = 1320000, 800000
    _pill(s, L, 1030000, 2400000, 250000, "ANTES · KERNEL EXACTO", AMBER)
    _card(s, x1, ya, w1, ha, [P("Datos", 14, True, TEXT)])
    _card(s, x2, ya, w2, ha, [P("StandardScaler", 14, True, TEXT)])
    _card(s, x3, ya, w3 + gap + w4, ha, [
        P("SVC / SVR con kernel RBF exacto", 14, True, TEXT, 3),
        P("Costo O(n²) a O(n³) con ~154.000 filas", 12, False, MUTED)],
        fill=PANEL_ALT, line=AMBER, line_w=2.0)
    for ax in (x1 + w1, x2 + w2):
        _arrow(s, ax + 25000, ya + ha // 2 - 110000, gap - 50000, 220000)
    _textbox(s, L, ya + ha + 40000, W, 280000, [P(
        "No termina: más de 39 h de CPU acumuladas; se canceló", 13, True, AMBER)])

    # Despues
    yb, hb = 2860000, 950000
    _pill(s, L, 2570000, 2400000, 250000, "DESPUÉS · SVM APROXIMADO", ACCENT)
    _card(s, x1, yb, w1, hb, [P("Datos", 14, True, TEXT)])
    _card(s, x2, yb, w2, hb, [P("StandardScaler", 14, True, TEXT)])
    _card(s, x3, yb, w3, hb, [
        P("Mapeo de Nystroem", 14, True, TEXT, 3),
        P("RBF, n_components", 12, False, MUTED)], fill=PANEL_ALT, line=ACCENT, line_w=2.0)
    _card(s, x4, yb, w4, hb, [
        P("", 12, runs=[("Clasificador: ", ACCENT, True),
                        ("SVM lineal (pérdida hinge) + calibración de Platt", TEXT, False)], after=4),
        P("", 12, runs=[("Regresor: ", ACCENT, True),
                        ("SVR lineal ε-insensible", TEXT, False)])],
        fill=PANEL_ALT, line=ACCENT, line_w=2.0)
    for ax in (x1 + w1, x2 + w2, x3 + w3):
        _arrow(s, ax + 25000, yb + hb // 2 - 110000, gap - 50000, 220000)
    _textbox(s, L, yb + hb + 40000, W, 280000, [P(
        "Costo lineal en n; se espera correr en segundos por fold (por medir)", 13, True, GREEN)])

    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, 4280000, W, 440000, fill=None, line=AMBER,
           line_w=2.0, paras=[P("SVM aproximado (kernel aproximado), no un SVM de kernel exacto",
                                14, True, AMBER)],
           margins=(150000, 30000, 150000, 30000))

    _notas(s, (
        "El SVM original usaba SVC y SVR exactos con kernel RBF, cuyo costo crece entre n al cuadrado y n al cubo; con unas 154.000 filas la corrida del 5 de octubre acumuló más de un día y 15 horas de CPU "
        "(más de 39 horas) sin terminar y se canceló. La alternativa es un SVM aproximado: se estandariza, se aproxima el kernel RBF con un mapeo de Nystroem de n_components componentes y sobre ese mapeo se entrena "
        "un clasificador lineal de margen con pérdida hinge, calibrado con Platt para obtener probabilidades, y un regresor lineal con pérdida ε-insensible (SVR lineal). "
        "El costo pasa a ser lineal en el número de filas; el tiempo real por fold lo vamos a medir en la corrida, hoy es una expectativa y no una cifra. "
        "Hay que decirlo con claridad: no es un SVM de kernel exacto, es un SVM aproximado, y así se documenta en el descriptor de preprocesado (svm_aproximado_nystroem).\n\n"
        "Fuentes: docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md §2 y §3.1; swmm_resilience/ml/bench/models/svm_family.py (docstring y build_classifier/build_regressor); "
        "docs/resultados/2026-10-06-handoff-svm.md §2 (más de 1 día y 15 h de CPU, cancelada, ~154.000 filas)."
    ))
    return s


# ---------------------------------------------------------------------------
# Diapositiva 6: comparacion justa
# ---------------------------------------------------------------------------
def slide_comparacion_justa(prs, con_resultados: bool):
    pie = FOOTER_TEXT + (
        "  ·  resultados en la siguiente diapositiva" if con_resultados
        else "  ·  resultados: corridas en curso")
    s = nueva_slide(
        prs, "Cómo se hará la comparación justa",
        "Tres familias con las mismas condiciones; solo cambia el modelo", pie)
    cw = 2750000
    xs = [L, L + cw + 150000, L + 2 * (cw + 150000)]
    familias = [
        ("XGBoost", "Árboles de gradiente", "Se presenta siempre", ACCENT),
        ("MLP", "Red neuronal", "Se presenta siempre", ACCENT),
        ("SVM aproximado", "Nystroem + SVM lineal", "Tercer candidato", AMBER),
    ]
    for x, (nombre, desc, etiqueta, color) in zip(xs, familias):
        _card(s, x, Y0 + 20000, cw, 930000, [
            P(nombre, 16, True, ACCENT, 2), P(desc, 12, False, MUTED, 2),
            P(etiqueta, 11, True, color)],
            fill=PANEL_ALT, line=color, line_w=2.0)
        _arrow(s, x + cw // 2 - 150000, Y0 + 990000, 300000, 230000, "down", LINE)

    filas = [
        ("Mismo dato", "tabular_v4_21 · 154.080 filas (prep_id ca706e27654e0cf5)"),
        ("Mismo presupuesto", "Mismo número de trials de Optuna y mismo CV interno por forma"),
        ("Mismo protocolo", "LeaveOneShapeOut: 40 pliegues, una forma retenida por pliegue"),
        ("Mismas métricas", "RMSE de volumen · F1 · NSE del regresor · matriz de confusión"),
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
           paras=[P("Estimación exploratoria condicionada: los hiperparámetros se eligen con todas las formas",
                    13, True, AMBER)],
           margins=(150000, 30000, 150000, 30000))

    _notas(s, (
        "Para comparar de forma justa las tres familias, todas usan el mismo dato (contrato tabular_v4_21, 154.080 filas), el mismo presupuesto de afinación con Optuna "
        "(mismo número de trials y el mismo CV interno por forma), el mismo protocolo de evaluación, LeaveOneShapeOut, y las mismas métricas. "
        "XGBoost y MLP se presentan siempre; el SVM aproximado entra como tercer candidato con el resultado que obtenga. "
        "La advertencia importante: los hiperparámetros se eligen usando todas las formas, así que la cifra de LeaveOneShapeOut es optimista por igual para las tres familias. "
        "Sirve para comparar entre ellas, no como confirmación independiente; por eso se declara como estimación exploratoria condicionada. "
        "Una afinación anidada por forma queda como trabajo futuro.\n\n"
        "Fuentes: docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md §1, §3.2 y §3.3 (decisiones del autor del 2026-10-12); "
        "04-informe-consolidado.md encabezado (prep_id ca706e27654e0cf5, 154.080 filas) y §5 punto 1."
    ))
    return s


# ---------------------------------------------------------------------------
# Diapositivas de resultados (solo si ya existen las tablas de la parte 2)
# ---------------------------------------------------------------------------
ETIQUETAS_MODELO = {
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "svm": "SVM aproximado",
    "baseline_interpolacion_factor": "Interpolación (ref.)",
    "baseline_forma_mas_cercana": "Forma más cercana (ref.)",
}
ORDEN_MODELO = ["xgboost", "mlp", "svm"]


def _etiqueta_modelo(modelo: str) -> str:
    if modelo in ETIQUETAS_MODELO:
        return ETIQUETAS_MODELO[modelo]
    return modelo.replace("_", " ").replace("svm", "SVM").replace("xgboost", "XGBoost").replace(
        "mlp", "MLP")


def _color_modelo(modelo: str) -> str:
    if modelo.startswith("baseline"):
        return AMBER
    if modelo.startswith("xgboost") or modelo.startswith("mlp"):
        return ACCENT
    if modelo.startswith("svm"):
        return LINE
    return MUTED


def _ordenar_modelos(rows: list[dict]) -> list[dict]:
    def clave(row):
        modelo = row["modelo"]
        if modelo.startswith("baseline"):
            return (2, modelo)
        for pos, nombre in enumerate(ORDEN_MODELO):
            if modelo == nombre:
                return (0, f"{pos}")
        for pos, nombre in enumerate(ORDEN_MODELO):
            if modelo.startswith(nombre):
                return (0, f"{pos}{modelo}")
        return (1, modelo)
    return sorted(rows, key=clave)


def agregar_slide_resultados(prs, datos_dir):
    """Agrega la slide de resultados (tabla + barras de RMSE y F1) y la de
    conclusiones/proximos pasos a partir de ``comparacion_protocolos.csv``
    de ``datos_dir`` (protocolo LeaveOneShapeOut). Devuelve las dos slides."""
    ruta = Path(datos_dir) / "comparacion_protocolos.csv"
    rows = [r for r in cargar_csv(ruta) if r["protocolo"] == "LeaveOneShapeOut"]
    if not rows:
        raise ValueError(f"{ruta} no tiene filas del protocolo LeaveOneShapeOut")
    rows = _ordenar_modelos(rows)
    modelos = [r["modelo"] for r in rows]
    etiquetas = [_etiqueta_modelo(m) for m in modelos]
    rmse = [round(float(r["rmse_vol_agregado_m3"]), 2) for r in rows]
    f1 = [round(float(r["f1_agregado"]), 4) for r in rows]
    nse = [float(r["nse_regresor_oracle"]) for r in rows]
    colores = [_color_modelo(m) for m in modelos]
    candidatos = [i for i, m in enumerate(modelos) if not m.startswith("baseline")]

    s = nueva_slide(
        prs, "Resultados: XGBoost, MLP y SVM aproximado",
        "LeaveOneShapeOut · mismos datos, presupuesto y métricas · estimación exploratoria condicionada")
    tabla = []
    for i, row in enumerate(rows):
        presentado = colores[i] == ACCENT
        color = ACCENT if presentado else TEXT
        tabla.append([[(etiquetas[i], color, presentado)],
                      [(fmt(rmse[i]), color, presentado)],
                      [(fmt(f1[i], 4), color, presentado)],
                      [(fmt(nse[i], 4), color, presentado)]])
    n = len(tabla)
    row_h = min(420000, 2900000 // max(n, 1))
    _tabla(s, L, Y0, [1900000, 800000, 750000, 800000],
           ["Modelo", "RMSE", "F1", "NSE"], tabla, row_h=row_h, header_h=340000, size=12,
           name="Table results")

    leyenda_y = Y0 + 340000 + row_h * n + 120000
    _textbox(s, L, leyenda_y, 4250000, 700000, [
        P("RMSE de volumen agregado (m³) · F1 agregado · NSE del regresor en nodos inundados",
          10, False, MUTED, 4),
        P("Azul: XGBoost y MLP (par presentado)", 10, True, ACCENT, 1),
        P("SVM aproximado: tercer candidato", 10, True, "7FA9D6", 1),
        P("Ámbar: referencias sin aprendizaje", 10, True, AMBER),
    ])

    cx = L + 4450000
    cw = W - 4450000
    _textbox(s, cx, Y0, cw, 230000, [P("RMSE de volumen (m³): menor es mejor", 11, False, MUTED)])
    _chart_barras(
        s, cx, Y0 + 220000, cw, 1620000, etiquetas,
        [("RMSE", rmse, ACCENT)], horizontal=True,
        etiquetas=[[fmt(v) for v in rmse]], colores_puntos=[colores],
        leyenda=False, size=11, gap=45)
    _textbox(s, cx, Y0 + 1900000, cw, 230000, [P("F1 agregado: mayor es mejor", 11, False, MUTED)])
    _chart_barras(
        s, cx, Y0 + 2120000, cw, 1620000, etiquetas,
        [("F1", f1, ACCENT)], horizontal=True,
        etiquetas=[[fmt(v, 4) for v in f1]], colores_puntos=[colores],
        leyenda=False, size=11, gap=45, maximo=1.4)

    mejor_rmse = min(candidatos, key=lambda i: rmse[i])
    mejor_f1 = max(candidatos, key=lambda i: f1[i])
    resumen = "; ".join(f"{etiquetas[i]} RMSE {fmt(rmse[i])} m³, F1 {fmt(f1[i], 4)}" for i in candidatos)
    _notas(s, (
        "Resultados del protocolo principal, LeaveOneShapeOut, con la misma base de datos, el mismo presupuesto de Optuna y las mismas métricas para las tres familias. "
        f"Menor RMSE de volumen entre los candidatos: {etiquetas[mejor_rmse]} ({fmt(rmse[mejor_rmse])} m³); mayor F1: {etiquetas[mejor_f1]} ({fmt(f1[mejor_f1], 4)}). "
        f"Detalle por candidato: {resumen}. "
        "Las barras ámbar son las referencias que no aprenden y sirven de piso. Estas cifras son una estimación exploratoria condicionada: los hiperparámetros se eligieron con todas las formas, "
        "así que son optimistas por igual para las tres familias y no llevan intervalos de confianza (40 formas).\n\n"
        f"Fuente: {ruta.as_posix()} (filas de LeaveOneShapeOut; columnas rmse_vol_agregado_m3, f1_agregado, nse_regresor_oracle); "
        "docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md §3.2-§3.3."
    ))

    # --- Conclusiones y proximos pasos ------------------------------------
    c = nueva_slide(
        prs, "Conclusiones y próximos pasos",
        "Lectura de la comparación y lo que queda por decidir")
    tw = 4150000
    tiles = [
        ("Menor RMSE de volumen", etiquetas[mejor_rmse], f"{fmt(rmse[mejor_rmse])} m³"),
        ("Mayor F1 agregado", etiquetas[mejor_f1], fmt(f1[mejor_f1], 4)),
    ]
    for i, (rotulo, nombre, valor) in enumerate(tiles):
        _card(c, L, Y0 + 20000 + i * 950000, tw, 880000, [
            P(rotulo, 12, False, MUTED, 1),
            P("", 20, runs=[(valor, ACCENT, True), ("   " + nombre, TEXT, False)])],
            fill=PANEL_ALT, line=ACCENT)
    _card(c, L, Y0 + 20000 + 2 * 950000, tw, 1230000, [
        P("Cómo leerlo", 12, True, AMBER, 3),
        P("Estimación exploratoria condicionada: hiperparámetros elegidos con todas las formas, "
          "sin intervalos de confianza", 12, False, TEXT)],
        fill=PANEL, line=AMBER, line_w=2.0)

    px = L + tw + 250000
    pw = W - tw - 250000
    pasos = [
        "Decidir la etiqueta: ruta A (recomendada) o B",
        "Documento 06: selección de modelos y texto de la tesis",
        "Afinación anidada por forma (trabajo futuro)",
        "Regenerar las figuras con el modelo declarado",
    ]
    _textbox(c, px, Y0 + 20000, pw, 260000, [P("PRÓXIMOS PASOS", 10, True, MUTED)])
    for i, paso in enumerate(pasos):
        py = Y0 + 330000 + i * 840000
        badge = _shape(c, MSO_SHAPE.OVAL, px, py + 90000, 420000, 420000, fill=ACCENT,
                       line=None, paras=[P(str(i + 1), 16, True, BG)], margins=(0, 0, 0, 0))
        badge.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        _card(c, px + 520000, py, pw - 520000, 600000, [P(paso, 13, False, TEXT)],
              fill=PANEL, line=LINE)

    _notas(c, (
        "Cierre: con el mismo dato, presupuesto de afinación y protocolo, "
        f"{etiquetas[mejor_rmse]} tiene el menor RMSE de volumen ({fmt(rmse[mejor_rmse])} m³) y {etiquetas[mejor_f1]} el mayor F1 ({fmt(f1[mejor_f1], 4)}) entre los candidatos. "
        "Hay que leerlo con la salvedad de siempre: son estimaciones exploratorias condicionadas, sin intervalos de confianza y con una sola red. "
        "Quedan cuatro pendientes: decidir la etiqueta (recomiendo la ruta A), escribir el documento 06 de selección de modelos, "
        "la afinación anidada por forma como trabajo futuro y regenerar las figuras con el modelo que se declare.\n\n"
        f"Fuentes: {ruta.as_posix()}; docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md §3.2-§3.3 y §5; "
        "docs/auditoria/correcciones/04-informe-consolidado.md §4, §5 y §6."
    ))
    return [s, c]


# ---------------------------------------------------------------------------
def construir(base: Path = BASE, salida: Path = OUTPUT, datos_dir: Path = DATOS_DIR,
              datos_parte2_dir: Path = DATOS_PARTE2_DIR) -> Presentation:
    base, salida = Path(base), Path(salida)
    if not base.exists():
        raise SystemExit(f"No existe {base}")
    if salida.resolve() == base.resolve():
        raise SystemExit("La salida no puede ser la presentacion base")
    prs = Presentation(str(base))
    con_resultados = (Path(datos_parte2_dir) / "comparacion_protocolos.csv").exists()

    slide_auditoria(prs, datos_dir)
    slide_protocolos(prs, datos_dir)
    slide_matriz(prs, datos_dir)
    slide_etiqueta(prs)
    slide_svm(prs)
    slide_comparacion_justa(prs, con_resultados)
    if con_resultados:
        agregar_slide_resultados(prs, datos_parte2_dir)

    prs.save(str(salida))
    return prs


def main() -> None:
    prs = construir()
    print(f"Guardado: {OUTPUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
