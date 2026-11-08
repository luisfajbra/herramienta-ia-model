"""Crea una presentación semanal con modelos conjuntos y escenarios nuevos."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pptx import Presentation
from pptx.opc.packuri import PackURI
from scripts.build_weekly_2026_11_08_joint_deck import slide, block, picture_slide
from scripts.update_weekly_2026_11_02_deck import (
    LIGHT_BLUE, SUBTITLE, WHITE, replace_text, renumber_slide_footers,
)

SOURCE = ROOT / 'Presentacion_MLSMFP_08-11_entrenamiento_conjunto_revision_auditoria.pptx'
SOURCE_SHA = 'cff5510b3543d01b69a8ce61dbfe31f706379b75d12817ef55d94bec6f43ced3'
OUTPUT = ROOT / 'Presentacion_MLSMFP_08-11_modelos_conjuntos_escenarios_nuevos.pptx'
FIGURES = ROOT / 'docs/figures_conjuntos_escenarios_nuevos_chico_norte_20261108'
DATA = ROOT / 'outputs/experiments/conjuntos_escenarios_nuevos_chico_norte_20261108/evaluacion_modelos_conjuntos'
SHAPES = [('pulso_agudo_recesion_6h', 'Pulso agudo'),
          ('ascenso_escalonado_recesion_12h', 'Ascenso escalonado')]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('La presentación de entrada cambió')
    manifest = json.loads((FIGURES / 'manifest.json').read_text(encoding='utf8'))
    scores_path = DATA / 'scores.json'
    if manifest['missing_maps'] or sha(scores_path) != manifest['scores_sha256']:
        raise ValueError('Resultados o mapas incompletos')
    for filename, expected in manifest['files'].items():
        if sha(FIGURES / filename) != expected:
            raise ValueError(f'La figura cambió: {filename}')
    scores = json.loads(scores_path.read_text(encoding='utf8'))
    prs = Presentation(SOURCE)
    if len(prs.slides) != 60:
        raise ValueError('La presentación de origen no tiene 60 diapositivas')
    # Remove the earlier Chico-Sur-only results from the copy, preserving them in SOURCE.
    for index in reversed(range(2, 41)):
        element = prs.slides._sldIdLst[index]
        prs.part.drop_rel(element.rId)
        prs.slides._sldIdLst.remove(element)
    # python-pptx assigns new part names based on the slide count. Renumber
    # retained parts so appending cannot collide with the old slide42+ names.
    for index, retained in enumerate(prs.slides, start=1):
        retained.part._partname = PackURI(f'/ppt/slides/slide{index}.xml')
    summary = prs.slides[1]
    x = scores['models']['xgboost']['overall']
    m = scores['models']['mlp']['overall']
    replacements = {
        1: ('Modelos conjuntos: evaluación de escenarios nuevos', 18),
        2: ('Chico Norte · dos hidrogramas y cinco factores nuevos.', 10),
        6: ('DATOS DE DESARROLLO', 10),
        7: ('1.332 corridas', 18),
        8: ('Entrenamiento con Chico Sur y Chico Norte.', 9),
        10: ('RMSE GLOBAL · NUEVOS ESCENARIOS', 9),
        11: (f"{x['rmse_m3']:.2f} / {m['rmse_m3']:.2f} m³".replace('.', ','), 17),
        12: ('XGBoost / MLP · 5.220 pares nodo–escenario.', 9),
        14: ('SIMULACIONES NUEVAS', 10),
        15: ('10 escenarios', 18),
        16: ('2 hidrogramas × 5 factores · 10 elegibles.', 9),
        18: ('ALCANCE', 10),
        19: ('Factores interpolados y formas nuevas.', 9),
        20: ('Chico Norte es una red ya conocida por los modelos.', 9),
    }
    for index, (value, size) in replacements.items():
        tone = LIGHT_BLUE if index in (1, 6, 10, 14, 18) else (SUBTITLE if index == 2 else WHITE)
        replace_text(summary.shapes[index], value, size=size, color_hex=tone,
                     bold=index in (1, 6, 7, 10, 11, 14, 15, 18))
        summary.shapes[index].line.fill.background()
    # Empty inherited placeholder renders as a black outline in PowerPoint.
    empty_placeholder = summary.shapes[0]
    if empty_placeholder.has_text_frame and not empty_placeholder.text.strip():
        empty_placeholder._element.getparent().remove(empty_placeholder._element)

    # Preserve the agreed visual order: hydrographs, single-model maps, parities, curves.
    for name, label in SHAPES:
        picture_slide(prs, 'Hidrograma de entrada', label + ' · Chico Norte',
                      FIGURES / f'hidrograma_{name}.png')
    for name, label in SHAPES:
        for factor in (1.9, 2.5, 3.7):
            for model, title in [('swmm', 'SWMM'), ('mlp', 'MLP'), ('xgboost', 'XGBoost')]:
                picture_slide(prs, 'Mapa de inundación', f'{label} · factor {factor:.1f} · {title}',
                              FIGURES / f'mapa_{name}_f{factor:.1f}_{model}.png')
    for model, label in [('mlp', 'MLP'), ('xgboost', 'XGBoost')]:
        picture_slide(prs, 'Paridad global', f'{label} · escenarios nuevos',
                      FIGURES / f'paridad_global_{model}.png')
    picture_slide(prs, 'Volumen total por factor', 'Chico Norte · comparación con SWMM',
                  FIGURES / 'volumen_total_por_factor.png')

    s = slide(prs, 'Resultados de la evaluación', 'Diez escenarios · 5.220 pares nodo–escenario.')
    block(s, .48, 'MLP', [f"RMSE: {m['rmse_m3']:.2f} m³".replace('.', ','),
        f"MAE: {m['mae_m3']:.2f} m³".replace('.', ','), f"Detección: {100*m['recall']:.2f} %".replace('.', ','),
        f"Sesgo de volumen total: {m['volume_bias_pct']:+.2f} %".replace('.', ',')])
    block(s, 5.14, 'XGBoost', [f"RMSE: {x['rmse_m3']:.2f} m³".replace('.', ','),
        f"MAE: {x['mae_m3']:.2f} m³".replace('.', ','), f"Detección: {100*x['recall']:.2f} %".replace('.', ','),
        f"Sesgo de volumen total: {x['volume_bias_pct']:+.2f} %".replace('.', ',')])
    s = slide(prs, 'Interpretación y próximos pasos', 'Evaluación frente a SWMM con modelos conjuntos.')
    block(s, .48, 'Hallazgos', ['MLP obtuvo menor error de volumen en este lote.',
        'XGBoost detectó más nodos inundados.',
        'El sesgo global de MLP combina errores que se compensan.',
        'El ascenso escalonado supera el rango de tiempo al pico.'])
    block(s, 5.14, 'Siguientes pasos', ['Revisar resultados por hidrograma y factor.',
        'Mantener el QA hidráulico provisional documentado.',
        'Evaluar transferencia en una tercera red no vista.',
        'Conservar simulaciones y resultados para la tesis.'])
    renumber_slide_footers(prs)
    if len(prs.slides) != 46:
        raise ValueError(f'Número inesperado de diapositivas: {len(prs.slides)}')
    prs.save(OUTPUT)
    with zipfile.ZipFile(OUTPUT) as z:
        names = z.namelist()
        if z.testzip() is not None or len(names) != len(set(names)):
            raise ValueError('Archivo PPTX inválido')
    check = Presentation(OUTPUT)
    if len(check.slides) != 46:
        raise ValueError('La copia no se puede verificar')
    print(json.dumps({'output': str(OUTPUT), 'slides': len(check.slides),
                      'sha256': sha(OUTPUT), 'scores_sha256': sha(scores_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
