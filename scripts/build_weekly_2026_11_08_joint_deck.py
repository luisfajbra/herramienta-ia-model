"""Append joint-model holdout results to the established weekly PowerPoint."""
from pathlib import Path
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches
from scripts.update_weekly_2026_11_02_deck import (
    add_slide, add_card, add_text, add_picture_contained, replace_text,
    renumber_slide_footers, NAVY, CARD_DARK, LIGHT_BLUE, SUBTITLE, WHITE,
)
from scripts.build_chico_norte_presentation_figures import plot_hydrograph_series

SOURCE = ROOT / 'Presentacion_MLSMFP_02-11_avance_chico_norte_mapas_individuales_por_modelo.pptx'
OUTPUT = ROOT / 'Presentacion_MLSMFP_08-11_entrenamiento_conjunto_y_reserva.pptx'
FIG = ROOT / 'docs/figures_reserva_conjunta_20261108_v2'
DIAG = ROOT / 'docs/figures_reserva_diagnostico_20261108'
HYDRO = ROOT / 'docs/figures_reserva_presentacion_20261108'
SCORES = ROOT / 'outputs/experiments/evaluacion_reserva_conjunta_20261108_v1/scores.json'


def slide(prs, title, subtitle):
    s = add_slide(prs, title, subtitle, len(prs.slides) + 1)
    for sh in s.shapes:
        if sh.has_text_frame and sh.text.startswith('MLSMFP | Avance:'):
            replace_text(sh, 'MLSMFP | Avance: entrenamiento conjunto | 08-11-2026', size=8, color_hex=SUBTITLE)
        elif sh.has_text_frame and sh.text == title:
            sh.top = Inches(.65); sh.height = Inches(.36)
            replace_text(sh, title, size=18, color_hex=LIGHT_BLUE, bold=True)
        elif sh.has_text_frame and sh.text == subtitle:
            sh.top = Inches(1.03); sh.height = Inches(.24)
            replace_text(sh, subtitle, size=9.2, color_hex=SUBTITLE)
    return s


def block(s, x, title, lines):
    add_card(s, x, 1.43, 4.38, 3.48, fill=CARD_DARK)
    add_text(s, title, x + .20, 1.58, 3.98, .32, size=13, color_hex=LIGHT_BLUE, bold=True)
    for i, line in enumerate(lines):
        add_text(s, '• ' + line, x + .22, 1.97 + i * .61, 3.93, .54, size=10.1, color_hex=WHITE)


def hydrographs():
    HYDRO.mkdir(parents=True, exist_ok=True)
    paths = {}
    for shape, label in [('common_storm', 'Evento de referencia'),
                         ('shape_quasi_uniform_block', 'Bloque de aporte sostenido')]:
        d = pd.read_csv(ROOT / 'data/hydrograph_shapes' / (shape + '.csv'))
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        plot_hydrograph_series(ax, d.time_h.to_numpy(), d.q_norm.to_numpy())
        ax.set_title('Hidrograma sintético de entrada', fontweight='bold', color='#172F4D')
        path = HYDRO / (shape + '.png')
        fig.savefig(path, dpi=200, facecolor='white'); plt.close(fig)
        paths[shape] = (path, label)
    return paths


def picture_slide(prs, title, subtitle, path):
    s = slide(prs, title, subtitle)
    add_picture_contained(s, path, .55, 1.35, 8.9, 3.73)


def repair_historical_layout(prs):
    """Keep historic content while clearing the inherited header on slides 3–41."""
    for s in list(prs.slides)[2:41]:
        for sh in list(s.shapes):
            top = sh.top / 914400
            text = sh.text_frame.text.strip() if sh.has_text_frame else ''
            if sh.shape_type == MSO_SHAPE_TYPE.PLACEHOLDER and not text:
                sh._element.getparent().remove(sh._element)
                continue
            if sh.shape_type == MSO_SHAPE_TYPE.TEXT_BOX:
                sh.line.fill.background()
            if text and top < .55:
                sh.top = Inches(.65); sh.height = Inches(.36)
                for p in sh.text_frame.paragraphs:
                    for run in p.runs:
                        if run.font.size:
                            run.font.size = min(run.font.size, Inches(18 / 72))
            elif text and .55 <= top < 1.02:
                sh.top = Inches(1.03); sh.height = Inches(.24)
                for p in sh.text_frame.paragraphs:
                    for run in p.runs:
                        if run.font.size:
                            run.font.size = min(run.font.size, Inches(9.2 / 72))
            elif 1.0 <= top < 5.2 and sh.shape_type != MSO_SHAPE_TYPE.PLACEHOLDER:
                old_width = sh.width
                sh.top = Inches(1.35 + (top - 1.02) * .9)
                sh.height = int(sh.height * .9)
                if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    sh.width = int(old_width * .9)
                    sh.left += int((old_width - sh.width) / 2)
                elif sh.has_text_frame:
                    for p in sh.text_frame.paragraphs:
                        for run in p.runs:
                            if run.font.size:
                                run.font.size = int(run.font.size * .9)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    status = json.loads((SCORES.parent / 'status.json').read_text(encoding='utf-8'))
    import hashlib
    with SCORES.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == status['scores_sha256']
    scores = json.loads(SCORES.read_text(encoding='utf-8'))
    prs = Presentation(SOURCE)
    if len(prs.slides) != 41:
        raise ValueError('unexpected source presentation')
    repair_historical_layout(prs)
    replace_text(prs.slides[0].shapes[0], '1')
    replace_text(prs.slides[0].shapes[3], '8 de noviembre de 2026')
    summary = prs.slides[1]
    replacements = {
        1: ('Entrenamiento conjunto y evaluación de formas reservadas', 18),
        2: ('Corte al 8 de noviembre · dos redes conocidas y siete formas reservadas.', 10),
        3: ('MLSMFP | Avance: entrenamiento conjunto | 08-11-2026', 8),
        4: ('2', 11),
        6: ('DATOS DE DESARROLLO', 10),
        7: ('1.332 corridas', 18),
        8: ('Chico Sur y Chico Norte; QA hidráulico provisional.', 9),
        10: ('RMSE MACRO · RESERVA', 10),
        11: ('234,54 / 184,63 m³', 17),
        12: ('XGBoost / MLP · promedio de RMSE de las dos redes.', 9),
        14: ('EVALUACIÓN RESERVADA', 10),
        15: ('321 escenarios', 18),
        16: ('Siete formas fuera del ajuste; 106.746 pares por modelo.', 9),
        18: ('MLP · ENSEMBLE', 10),
        19: ('Cinco semillas ajustadas y verificadas.', 9),
        20: ('Redes conocidas; falta comprobar transferencia a una tercera red.', 9),
    }
    for index, (value, size) in replacements.items():
        tone = LIGHT_BLUE if index in (1, 4, 6, 10, 14, 18) else (SUBTITLE if index in (2, 3) else WHITE)
        replace_text(summary.shapes[index], value, size=size, color_hex=tone, bold=index in (1, 6, 7, 10, 11, 14, 15, 18))
        summary.shapes[index].line.fill.background()
    summary.shapes[1].top = Inches(.65)
    summary.shapes[1].height = Inches(.36)
    summary.shapes[2].top = Inches(1.03)
    summary.shapes[2].height = Inches(.24)
    paths = hydrographs()
    s = slide(prs, 'Entrenamiento conjunto', 'Nueva etapa: un mismo modelo aprende con Chico Sur y Chico Norte.')
    block(s, .48, 'Datos de desarrollo', ['866 corridas elegibles de Chico Sur y 787 de Chico Norte.',
        '1.332 corridas para ajustar; QA hidráulico provisional.',
        'XGBoost y MLP: 50 pruebas de hiperparámetros cada uno.',
        'MLP final: ensemble de cinco semillas.'])
    block(s, 5.14, 'Evaluación reservada', ['Siete formas completas quedaron fuera del entrenamiento.',
        '321 corridas: 168 de Chico Sur y 153 de Chico Norte.',
        '106.746 pares nodo–escenario para cada modelo.',
        'Ambas redes son conocidas; falta evaluar una red nueva.'])
    for shape in ('common_storm', 'shape_quasi_uniform_block'):
        path, label = paths[shape]
        picture_slide(prs, 'Hidrograma de entrada', label + ' · forma sintética normalizada.', path)
    for name, network in [('Chico Sur', 'chico_sur'), ('Chico Norte', 'chico_norte')]:
        for model in ('SWMM', 'MLP', 'XGBoost'):
            picture_slide(prs, 'Mapa de inundación', name + ' · evento de referencia · factor 3 · ' + model,
                          FIG / f'mapa_{network}_{model.lower()}_factor_3.png')
    for family, label in [('xgboost', 'XGBoost'), ('mlp', 'MLP')]:
        picture_slide(prs, 'Paridad global', label + ' · comparación con SWMM en las formas reservadas.',
                      FIG / ('paridad_global_' + family + '.png'))
    for network, name in [('chico_sur', 'Chico Sur'), ('chico_norte', 'Chico Norte')]:
        picture_slide(prs, 'Volumen total por factor', name + ' · diagnóstico del hidrograma de aporte sostenido.',
                      DIAG / ('volumen_factor_' + network + '.png'))
    s = slide(prs, 'Resultados de la reserva', 'Comparación por red sobre las formas excluidas del ajuste.')
    for i, (family, label) in enumerate([('xgboost', 'XGBoost'), ('mlp', 'MLP')]):
        m = scores['models'][family]; a, b = m['by_network']['1'], m['by_network']['2']
        block(s, .48 if i == 0 else 5.14, label, [
            f"RMSE macro entre redes: {m['macro_network_rmse_m3']:.2f} m³.",
            f"RMSE Chico Sur / Chico Norte: {a['rmse_m3']:.2f} / {b['rmse_m3']:.2f} m³.",
            f"Recall Chico Sur / Chico Norte: {100*a['recall']:.1f} % / {100*b['recall']:.1f} %.",
            f"Sesgo de volumen: {a['volume_bias_pct']:.1f} % / {b['volume_bias_pct']:.1f} %."])
    s = slide(prs, 'Hallazgos y próximos pasos', 'Resultados frente a SWMM; las redes siguen bajo los límites hidráulicos documentados.')
    block(s, .48, 'Hallazgos', ['MLP logró menor RMSE de volumen en esta reserva.',
        'XGBoost detectó más nodos inundados.',
        'Un hidrograma sostenido concentra gran parte del error cuadrático.',
        'El factor 3 de los mapas se eligió después de ver los resultados.'])
    block(s, 5.14, 'Siguientes pasos', ['Revisar hidráulicamente el hidrograma con mayor error.',
        'Revisar cobertura de entrenamiento y respuesta por factor.',
        'Evaluar transferencia en una tercera red no vista.',
        'Mantener trazabilidad, QA y resultados para el documento final.'])
    s = slide(prs, 'Revisión del hidrograma', 'Diagnóstico descriptivo realizado después de la evaluación.')
    block(s, .48, 'Forma de entrada', ['Duración: 11 horas.',
        'Permanece unas 8,24 horas al 90 % o más del caudal pico.',
        'Su volumen normalizado equivale a 9,57 horas al pico.',
        'Es un hidrograma sintético; no representa por sí solo una lluvia observada.'])
    block(s, 5.14, 'Revisión de las corridas', ['Se revisaron factores 2, 3, 4 y 5 en ambas redes.',
        'Todos pasaron el QA provisional y se reportaron drenados.',
        'Continuidad entre 0,042 % y 0,243 % en esos casos.',
        'El origen de los caudales iniciales de Chico Norte sigue sin aclarar.'])
    s = slide(prs, 'Cobertura del entrenamiento', 'Comparación por nodo y red con los valores máximos usados en el ajuste.')
    block(s, .48, 'Factor 5', ['El caudal pico no superó el máximo de entrenamiento en los nodos revisados.',
        'El volumen tributario sí superó ese máximo en los 160 nodos de Chico Sur y los 522 de Chico Norte.',
        'En factores 2, 3 y 4 no superó ese máximo para esta variable.',
        'Esto describe la cobertura; no identifica por sí solo la causa del error.'])
    block(s, 5.14, 'Interpretación', ['La diferencia en volumen puede importar aunque el pico esté dentro del rango.',
        'Un volumen total cercano puede ocultar errores en nodos individuales.',
        'La reserva ya fue observada: no debe reutilizarse como prueba ciega tras nuevos ajustes.',
        'El siguiente paso es revisar el dominio hidráulico y buscar una red independiente.'])
    renumber_slide_footers(prs)
    prs.save(OUTPUT)
    check = Presentation(OUTPUT)
    if len(check.slides) != 58:
        raise ValueError(f'unexpected slide count: {len(check.slides)}')
    with zipfile.ZipFile(OUTPUT) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None:
            raise ValueError('invalid PowerPoint package')
    print(json.dumps({'output': str(OUTPUT), 'slides': len(check.slides), 'new_slides': 17}, ensure_ascii=False))


if __name__ == '__main__':
    main()
