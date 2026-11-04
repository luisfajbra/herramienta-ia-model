"""Update the approved weekly deck with separated Chico Norte figures."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Presentacion_MLSMFP_30-10_avance_chico_norte_y_modelos_paridad_global_final.pptx"
OUTPUT = ROOT / "Presentacion_MLSMFP_02-11_avance_chico_norte_mapas_separados_inundacion.pptx"
TEMP_OUTPUT = OUTPUT.with_name(OUTPUT.stem + ".building.pptx")
RUN = ROOT / "outputs/experiments/chico_norte_novel_hydrograph_eval_20261102/run_20261103T034108Z"
FIGURES = ROOT / "docs/figures_chico_norte_novel_eval_2026_11_02_presentacion"

NAVY, CARD, CARD_DARK = "0E2A47", "234560", "193A58"
LIGHT_BLUE, SUBTITLE, WHITE = "8BC5FA", "B7CBE0", "F7FAFE"
ORANGE, BLUE, PALE = "E87525", "4386C6", "E9F2FA"


def color(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def replace_text(shape, value: str, *, size: float | None = None,
                 color_hex: str | None = None, bold: bool | None = None) -> None:
    tf = shape.text_frame
    old = tf.paragraphs[0] if tf.paragraphs else None
    first = old.runs[0] if old and old.runs else None
    font_name = first.font.name if first else "Montserrat"
    font_size = first.font.size.pt if first and first.font.size else 10
    alignment = old.alignment if old else PP_ALIGN.LEFT
    tf.clear()
    paragraph = tf.paragraphs[0]
    paragraph.alignment = alignment
    run = paragraph.add_run()
    run.text = value
    run.font.name = font_name or "Montserrat"
    run.font.size = Pt(size if size is not None else font_size)
    run.font.bold = bold if bold is not None else (first.font.bold if first else False)
    if color_hex:
        run.font.color.rgb = color(color_hex)
    tf.word_wrap = True


def add_text(slide, value: str, x: float, y: float, w: float, h: float,
             *, size: float = 11, color_hex: str = WHITE,
             font: str = "Montserrat", bold: bool = False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.03)
    frame.margin_right = Inches(0.03)
    frame.margin_top = Inches(0.015)
    frame.margin_bottom = Inches(0.015)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color(color_hex)
    return shape


def add_card(slide, x: float, y: float, w: float, h: float,
             *, fill: str = CARD, line: str = PALE) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    shape.line.color.rgb = color(line)
    shape.line.width = Pt(0.8)


def add_picture_contained(slide, path: Path, x: float, y: float,
                          max_w: float, max_h: float) -> None:
    with Image.open(path) as image:
        ratio = image.width / image.height
    width, height = max_w, max_w / ratio
    if height > max_h:
        height, width = max_h, max_h * ratio
    slide.shapes.add_picture(str(path), Inches(x + (max_w - width) / 2),
                             Inches(y + (max_h - height) / 2),
                             width=Inches(width), height=Inches(height))


def add_slide(prs, title: str, subtitle: str, page: int):
    layout = next(layout for layout in prs.slide_layouts if layout.name == "Title only")
    slide = prs.slides.add_slide(layout)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(NAVY)
    add_text(slide, title, 0.44, 0.14, 9.12, 0.48, size=21.5,
             color_hex=LIGHT_BLUE, font="Libre Baskerville", bold=True)
    add_text(slide, subtitle, 0.47, 0.63, 9.05, 0.38, size=10, color_hex=SUBTITLE)
    add_text(slide, "MLSMFP | Avance: evaluación externa Chico Norte | 02-11-2026",
             0.34, 5.31, 8.6, 0.18, size=8, color_hex=SUBTITLE)
    add_text(slide, str(page), 9.1, 5.25, 0.55, 0.25, size=11,
             color_hex=LIGHT_BLUE, align=PP_ALIGN.CENTER)
    return slide


def update_existing_summary(prs) -> None:
    for shape in prs.slides[0].shapes:
        if getattr(shape, "has_text_frame", False) and "30 de octubre de 2026" in shape.text:
            replace_text(shape, "2 de noviembre de 2026")
    for shape in prs.slides[1].shapes:
        if not getattr(shape, "has_text_frame", False):
            continue
        text = shape.text.strip()
        if text.startswith("Corte al 30 de octubre"):
            replace_text(shape, "Corte al 2 de noviembre · candidatos congelados; dos hidrogramas nuevos evaluados frente a SWMM.")
        elif text.startswith("CHICO NORTE · CAMPAÑA SWMM"):
            replace_text(shape, "CAMPAÑA SWMM ANTERIOR")
        elif text.startswith("Chico Norte se mantuvo fuera del ajuste"):
            replace_text(shape, "Nuevo piloto: 10/10 escenarios QA aptos; 5.220 pares nodo–escenario. Sin reentrenar los modelos.", size=9.1)
        elif "30-10-2026" in text and "MLSMFP | Avance" in text:
            replace_text(shape, text.replace("30-10-2026", "02-11-2026"))


def remove_inherited_comparison_map_slides(prs) -> list[int]:
    """Drop obsolete Chico Norte comparison maps that contain FP/FN markers."""
    target_indices = (12, 13, 14)
    if len(prs.slides) != 17:
        raise ValueError("the approved source deck must have 17 slides before cleanup")
    target_text = []
    for index in target_indices:
        slide = prs.slides[index]
        text = " ".join(shape.text for shape in slide.shapes
                        if getattr(shape, "has_text_frame", False))
        if "Chico Norte: escenario" not in text or "XGB FP" not in text or "FN" not in text:
            raise ValueError(f"source slide {index + 1} is not the obsolete error-marked map slide")
        target_text.append(text)
    slide_ids = prs.slides._sldIdLst
    for index in reversed(target_indices):
        slide_id = slide_ids[index]
        prs.part.drop_rel(slide_id.rId)
        slide_ids.remove(slide_id)
    return [index + 1 for index in target_indices]


def renumber_slide_footers(prs) -> None:
    """Keep numeric footer labels aligned with the final, edited slide order."""
    for page, slide in enumerate(prs.slides, start=1):
        matches = [shape for shape in slide.shapes
                   if getattr(shape, "has_text_frame", False)
                   and shape.text.strip().isdigit()
                   and shape.left >= Inches(8.8) and shape.top >= Inches(5.0)]
        if len(matches) == 1:
            replace_text(matches[0], str(page), size=11, color_hex=LIGHT_BLUE)


def repair_slide_part_collisions(path: Path, *, retained_slides: int = 14) -> None:
    """Assign unique OOXML part names to appended slides after source cleanup."""
    ns_p = "http://schemas.openxmlformats.org/presentationml/2006/main"
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ns_rel = "http://schemas.openxmlformats.org/package/2006/relationships"
    ns_ct = "http://schemas.openxmlformats.org/package/2006/content-types"
    ET.register_namespace("", ns_rel)
    ET.register_namespace("", ns_ct)
    ET.register_namespace("p", ns_p)
    ET.register_namespace("r", ns_r)

    with zipfile.ZipFile(path, "r") as source:
        infos = source.infolist()
        presentation = ET.fromstring(source.read("ppt/presentation.xml"))
        rels = ET.fromstring(source.read("ppt/_rels/presentation.xml.rels"))
        rel_by_id = {rel.attrib["Id"]: rel for rel in rels}
        slide_ids = presentation.find(f"{{{ns_p}}}sldIdLst")
        if slide_ids is None or len(slide_ids) != 38:
            raise ValueError("expected 38 ordered slides before repairing slide parts")
        remaps = {}
        for index, slide_id in enumerate(list(slide_ids)[retained_slides:], start=39):
            rel = rel_by_id[slide_id.attrib[f"{{{ns_r}}}id"]]
            old_target = rel.attrib["Target"]
            if not re.fullmatch(r"slides/slide\d+\.xml", old_target):
                raise ValueError(f"unexpected slide target: {old_target}")
            new_target = f"slides/slide{index}.xml"
            rel.attrib["Target"] = new_target
            remaps[old_target] = new_target

        counts = {name: sum(info.filename == name for info in infos)
                  for name in {info.filename for info in infos}}
        seen = {}
        output = []
        for info in infos:
            name = info.filename
            replacement = name
            match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
            rel_match = re.fullmatch(r"ppt/slides/_rels/slide(\d+)\.xml\.rels", name)
            if match:
                old_target = f"slides/slide{match.group(1)}.xml"
                if old_target in remaps:
                    occurrence = seen.get(name, 0)
                    seen[name] = occurrence + 1
                    if counts[name] == 1 or occurrence > 0:
                        replacement = "ppt/" + remaps[old_target]
            elif rel_match:
                old_target = f"slides/slide{rel_match.group(1)}.xml"
                if old_target in remaps:
                    occurrence = seen.get(name, 0)
                    seen[name] = occurrence + 1
                    if counts[name] == 1 or occurrence > 0:
                        replacement = "ppt/slides/_rels/" + remaps[old_target].split("/")[-1] + ".rels"
            output.append((info, replacement, source.read(info)))

        content_types = ET.fromstring(source.read("[Content_Types].xml"))
        present = {node.attrib.get("PartName") for node in content_types}
        for target in remaps.values():
            part_name = "/ppt/" + target
            if part_name not in present:
                ET.SubElement(content_types, f"{{{ns_ct}}}Override", {
                    "PartName": part_name,
                    "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
                })

        repaired = path.with_name(path.stem + ".repaired.pptx")
        with zipfile.ZipFile(repaired, "w", compression=zipfile.ZIP_DEFLATED) as dest:
            for info, name, payload in output:
                if info.filename == "ppt/presentation.xml":
                    payload = ET.tostring(presentation, encoding="utf-8", xml_declaration=True)
                elif info.filename == "ppt/_rels/presentation.xml.rels":
                    payload = ET.tostring(rels, encoding="utf-8", xml_declaration=True)
                elif info.filename == "[Content_Types].xml":
                    payload = ET.tostring(content_types, encoding="utf-8", xml_declaration=True)
                dest.writestr(name, payload)
    repaired.replace(path)


def add_hydrograph_slide(prs, *, page: int, shape_id: str,
                         label: str, factors: list[float]) -> None:
    title = "Hidrograma de pulsos dobles" if "cascade" in shape_id else "Hidrograma de meseta sostenida"
    slide = add_slide(prs, title,
                      "Forma normalizada de entrada · curva con el estilo del hidrograma de la corrida completa.", page)
    add_picture_contained(slide, FIGURES / f"hidrograma_{shape_id}.png", 0.55, 1.06, 8.9, 3.70)
    add_card(slide, 0.59, 4.82, 8.82, 0.40, fill=CARD_DARK)
    factor_text = " · ".join(f"{value:.1f}" for value in factors)
    add_text(slide, f"{label}  ·  factores simulados: {factor_text}",
             0.74, 4.87, 8.52, 0.22, size=9.3, bold=True, align=PP_ALIGN.CENTER)


def add_map_slide(prs, *, page: int, map_record: dict) -> None:
    shape_name = "Pulsos dobles" if "cascade_peaks" in map_record["shape_id"] else "Meseta sostenida y pico tardío"
    slide = add_slide(prs, "Mapa de inundación",
                      f"{shape_name} · factor {map_record['factor_mult']:.1f} · modelo {map_record['model_name']}", page)
    add_picture_contained(slide, FIGURES / map_record["filename"], 0.58, 1.02, 8.84, 4.16)


def add_parity_slide(prs, *, page: int, family: str,
                     results: dict, figure_metrics: dict) -> None:
    metric = results["models"][family]["overall"]
    parity = figure_metrics[family]
    label = "XGBoost" if family == "xgboost" else "MLP · ensemble de 5 semillas"
    model_color = BLUE if family == "xgboost" else ORANGE
    slide = add_slide(prs, "Paridad global",
                      f"{label} · cada punto representa un nodo en uno de los diez escenarios elegibles.", page)
    add_picture_contained(slide, FIGURES / f"paridad_global_{family}.png", 0.22, 1.02, 6.30, 3.72)
    add_card(slide, 6.72, 1.35, 2.75, 3.10)
    add_text(slide, label, 6.95, 1.55, 2.32, 0.32, size=13, color_hex=model_color, bold=True)
    add_text(slide, f"Precisión  {metric['precision']*100:.1f}%\nRecall  {metric['recall']*100:.1f}%\n"
             f"F1  {metric['f1']:.3f}\nPR-AUC  {metric['pr_auc']:.3f}",
             6.96, 2.02, 2.31, 1.23, size=10.2, bold=True)
    add_text(slide, f"RMSE  {parity['rmse_m3']:.2f} m³\nMAE  {parity['mae_m3']:.2f} m³\n"
             f"Omitidas  {metric['confusion_matrix']['fn']}\nFalsas alarmas  {metric['confusion_matrix']['fp']}",
             6.96, 3.34, 2.31, 0.90, size=9.4)
    add_text(slide, "RMSE agrupado por nodo–escenario; los ejes logarítmicos muestran ceros y volúmenes grandes.",
             0.54, 5.00, 8.90, 0.21, size=8.0, color_hex=SUBTITLE, align=PP_ALIGN.CENTER)


def add_factor_lines_slide(prs, *, page: int) -> None:
    slide = add_slide(prs, "Volumen total por factor",
                      "Comparación de SWMM y ambos modelos en los cinco factores de cada forma.", page)
    add_picture_contained(slide, FIGURES / "volumen_total_por_factor.png", 0.48, 1.08, 9.04, 3.82)
    add_text(slide, "Cada recta corresponde a una forma de hidrograma; cada punto, a un factor evaluado.",
             0.7, 4.91, 8.6, 0.24, size=9.0, color_hex=WHITE, align=PP_ALIGN.CENTER)


def add_conclusion_slide(prs, *, page: int) -> None:
    slide = add_slide(prs, "Qué nos dicen estos resultados",
                     "Es una prueba externa inicial; no se modificaron ni reentrenaron los candidatos.", page)
    add_card(slide, 0.48, 1.25, 4.39, 2.72)
    add_text(slide, "Hallazgos", 0.69, 1.39, 3.95, 0.3,
             size=12.5, color_hex=LIGHT_BLUE, bold=True)
    bullets = [
        "10 de 10 corridas pasaron la QA vigente; el límite de continuidad sigue provisional.",
        "MLP: recall 89,6%; menos inundaciones omitidas, con más falsas alarmas.",
        "XGBoost: precisión 82,2%; menos falsas alarmas, con más omisiones.",
        "RMSE casi igual: 186,26 m³ en MLP y 186,18 m³ en XGBoost.",
    ]
    for index, text in enumerate(bullets):
        add_text(slide, "• " + text, 0.7, 1.83 + index * 0.48, 3.94, 0.42, size=8.7)
    add_card(slide, 5.13, 1.25, 4.39, 2.72, fill=CARD_DARK, line=PALE)
    add_text(slide, "Siguiente", 5.34, 1.39, 3.95, 0.3, size=12.5, color_hex=WHITE, bold=True)
    next_steps = [
        "Revisar los nodos donde discrepan SWMM, MLP y XGBoost.",
        "Ampliar formas de hidrograma y factores; mantener Chico Norte fuera del ajuste por ahora.",
        "Repetir en más redes antes de afirmar generalización amplia.",
        "Justificar QA y revisar no convergencia antes de admitir etiquetas al entrenamiento.",
    ]
    for index, text in enumerate(next_steps):
        add_text(slide, "• " + text, 5.34, 1.83 + index * 0.48, 3.94, 0.42, size=8.7)
    add_card(slide, 0.57, 4.2, 8.86, 0.72, fill=CARD)
    add_text(slide, "Diez escenarios sintéticos describen este lote; todavía no demuestran confiabilidad en otras redes.",
             0.76, 4.31, 8.47, 0.48, size=9.1, bold=True, align=PP_ALIGN.CENTER)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path in (SOURCE, RUN / "evaluation_results.json", FIGURES / "presentation_figures_manifest.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    results = json.loads((RUN / "evaluation_results.json").read_text(encoding="utf-8"))
    completion = json.loads((RUN / "evaluation_completion.json").read_text(encoding="utf-8"))
    presentation = json.loads((FIGURES / "presentation_figures_manifest.json").read_text(encoding="utf-8"))
    if completion["status"] != "complete" or results["eligible_scenarios"] != 10:
        raise ValueError("The frozen evaluation must be complete before building the deck")
    if presentation["source_evaluation_results_sha256"] != _sha(RUN / "evaluation_results.json"):
        raise ValueError("presentation figures do not match the frozen score source")
    expected_figures = set(presentation["figure_sha256"])
    if len(presentation["maps"]) != 18 or len(presentation["parity_metrics"]) != 2:
        raise ValueError("expected 18 individual model maps and two separate model parity plots")
    if len(expected_figures) != 23:
        raise ValueError(f"expected 23 standalone presentation figures, found {len(expected_figures)}")
    for filename, digest in presentation["figure_sha256"].items():
        if _sha(FIGURES / filename) != digest:
            raise ValueError(f"presentation figure hash mismatch: {filename}")

    prs = Presentation(SOURCE)
    if len(prs.slides) != 17:
        raise ValueError(f"Unexpected reference deck slide count: {len(prs.slides)}")
    update_existing_summary(prs)
    removed_slides = remove_inherited_comparison_map_slides(prs)
    design = json.loads((RUN / "evaluation_design.json").read_text(encoding="utf-8"))
    factors = design["factor_selection"]["sorted_for_reporting"]
    add_hydrograph_slide(prs, page=15, shape_id="novel_20261102_cascade_peaks",
                         label="Pulsos dobles separados", factors=factors)
    add_hydrograph_slide(prs, page=16, shape_id="novel_20261102_sustained_late_surge",
                         label="Meseta sostenida y pico tardío", factors=factors)
    for page, map_record in enumerate(presentation["maps"], start=17):
        add_map_slide(prs, page=page, map_record=map_record)
    add_parity_slide(prs, page=35, family="xgboost", results=results,
                     figure_metrics=presentation["parity_metrics"])
    add_parity_slide(prs, page=36, family="mlp", results=results,
                     figure_metrics=presentation["parity_metrics"])
    add_factor_lines_slide(prs, page=37)
    add_conclusion_slide(prs, page=38)
    renumber_slide_footers(prs)
    if len(prs.slides) != 38:
        raise ValueError(f"Expected 38 slides after update, found {len(prs.slides)}")
    prs.save(TEMP_OUTPUT)
    repair_slide_part_collisions(TEMP_OUTPUT)
    check = Presentation(TEMP_OUTPUT)
    if len(check.slides) != 38:
        raise ValueError("saved deck did not retain all 38 slides")
    TEMP_OUTPUT.replace(OUTPUT)
    print(f"Saved {OUTPUT} with {len(check.slides)} slides; removed inherited slides {removed_slides}; order: hydrographs, individual model maps, parity, factor lines")


if __name__ == "__main__":
    main()
