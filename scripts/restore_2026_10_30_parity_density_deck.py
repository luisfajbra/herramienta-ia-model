"""Restore the original density parity plots in the 2026-10-30 deck."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Presentacion_MLSMFP_30-10_avance_chico_norte_y_modelos_con_graficas.pptx"
OUTPUT = ROOT / "Presentacion_MLSMFP_30-10_avance_chico_norte_y_modelos_graficas_paridad_densidad.pptx"
FIGURES = ROOT / "docs" / "figures_chico_norte_transferencia" / "estilo_corrida_completa"
MANIFEST = FIGURES / "figure_manifest.json"


def set_text(shape, value: str) -> None:
    tf = shape.text_frame
    old = tf.paragraphs[0] if tf.paragraphs else None
    old_run = old.runs[0] if old and old.runs else None
    font_name = old_run.font.name if old_run else None
    font_size = old_run.font.size if old_run else None
    font_bold = old_run.font.bold if old_run else None
    font_color = None
    if old_run:
        try:
            font_color = old_run.font.color.rgb
        except (AttributeError, TypeError):
            pass
    tf.clear()
    p = tf.paragraphs[0]
    if old:
        p.alignment = old.alignment
    run = p.add_run()
    run.text = value
    if font_name:
        run.font.name = font_name
    if font_size:
        run.font.size = font_size
    if font_bold is not None:
        run.font.bold = font_bold
    if font_color:
        run.font.color.rgb = font_color
    tf.word_wrap = True


def image_bytes(slide) -> bytes:
    pictures = [shape for shape in slide.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
    if len(pictures) != 1:
        raise ValueError(f"Expected one parity image, found {len(pictures)}")
    return pictures[0].image.blob


def main() -> None:
    for path in (SOURCE, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing presentation: {OUTPUT}")

    prs = Presentation(SOURCE)
    if len(prs.slides) != 18:
        raise ValueError(f"Unexpected source slide count: {len(prs.slides)}")

    all_pairs = image_bytes(prs.slides[11])
    flooded = image_bytes(prs.slides[12])
    all_path = FIGURES / "paridad_transferencia_chico_norte.png"
    flooded_path = FIGURES / "paridad_nodos_inundados_chico_norte.png"
    all_path.write_bytes(all_pairs)
    flooded_path.write_bytes(flooded)

    set_text(
        prs.slides[11].shapes[3],
        "Cada panel compara SWMM con el modelo. La diagonal roja y = x representa igualdad: volumen predicho = volumen SWMM.",
    )
    set_text(
        prs.slides[12].shapes[3],
        "Solo nodos inundados por SWMM (?1 m?). Vista condicional; sus RMSE no sustituyen la m?trica global.",
    )
    prs.save(OUTPUT)

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["parity"]["style"] = "two-panel hexbin density, log-count color scale, log10(1+volume) axes, dashed y=x equality line; restored from previous complete-run deck"
    payload["parity"]["all_pairs"]["figure"] = all_path.name
    payload["parity"]["swmm_flooded_subset"]["figure"] = flooded_path.name
    payload["restored_parity_source_presentation"] = {
        "filename": SOURCE.name,
        "sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "slide_all_pairs": 12,
        "slide_swmm_flooded_subset": 13,
    }
    for item in payload["artifacts"]:
        if item["path"] == all_path.name:
            path = all_path
        elif item["path"] == flooded_path.name:
            path = flooded_path
        else:
            continue
        data = path.read_bytes()
        item["sha256"] = sha256(data).hexdigest()
        item["byte_size"] = len(data)
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Restored previous density parity plots in {OUTPUT}")
    print(f"Global image SHA-256: {sha256(all_pairs).hexdigest()}")
    print(f"Flooded-only image SHA-256: {sha256(flooded).hexdigest()}")


if __name__ == "__main__":
    main()
