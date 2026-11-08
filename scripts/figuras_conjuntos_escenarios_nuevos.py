"""Figuras de los escenarios nuevos con modelos conjuntos y escala común por caso."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts import build_chico_norte_presentation_figures as old_figures
from scripts.figuras_reserva_conjunta_20261108 import draw_map, draw_parity

RUN = ROOT / 'outputs/experiments/conjuntos_escenarios_nuevos_chico_norte_20261108'
DATA = RUN / 'evaluacion_modelos_conjuntos'
OUTPUT = ROOT / 'docs/figures_conjuntos_escenarios_nuevos_chico_norte_20261108'
SHAPES = ROOT / 'data/hydrograph_validation/chico_norte_conjuntos_nuevos_20261108'
INP = ROOT / 'data/networks/chico_norte_93_qx1/SWMM - Chico Norte 93% - Qx1.00.inp'
LABELS = {'pulso_agudo_recesion_6h': 'Pulso agudo y recesión',
          'ascenso_escalonado_recesion_12h': 'Ascenso escalonado y recesión'}
MAP_FACTORS = (1.9, 2.5, 3.7)
KEYS = ['run_id', 'node_id']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    scores = json.loads((DATA / 'scores.json').read_text(encoding='utf8'))
    if scores['status'] != 'evaluacion_completa' or sha(DATA / 'prediction_seal.json') != scores['prediction_seal_sha256']:
        raise ValueError('La evaluación no está completa o el sello cambió')
    seal = json.loads((DATA / 'prediction_seal.json').read_text(encoding='utf8'))
    if sha(DATA / 'truth.parquet') != scores['truth_sha256']:
        raise ValueError('Las etiquetas cambiaron')
    truth = pd.read_parquet(DATA / 'truth.parquet')
    frames = {}
    for model in ('xgboost', 'mlp'):
        path = DATA / f'{model}_predictions.parquet'
        if sha(path) != seal['models'][model]['predictions_sha256']:
            raise ValueError('Las predicciones cambiaron')
        frames[model] = truth.merge(pd.read_parquet(path), on=KEYS, validate='one_to_one')
        if len(frames[model]) != len(truth):
            raise ValueError('Las predicciones no cubren las etiquetas')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    old_figures.OUTPUT, old_figures.SHAPE_LABELS = OUTPUT, LABELS
    for name in LABELS:
        old_figures.draw_hydrograph(SHAPES / f'{name}.csv', name)
    for model, title in [('xgboost', 'XGBoost'), ('mlp', 'MLP')]:
        if truth.vol_inundacion_m3.nunique() > 1:
            draw_parity(frames[model], title, OUTPUT / f'paridad_global_{model}.png')
    missing = []
    for name in LABELS:
        for factor in MAP_FACTORS:
            a = frames['xgboost'].loc[(frames['xgboost'].shape_id == name) & np.isclose(frames['xgboost'].factor_mult, factor)].sort_values('node_id')
            b = frames['mlp'].loc[(frames['mlp'].shape_id == name) & np.isclose(frames['mlp'].factor_mult, factor)].sort_values('node_id')
            if a.empty and b.empty:
                missing.append({'shape_id': name, 'factor': factor})
                continue
            if a.node_id.tolist() != b.node_id.tolist() or len(a) != 522:
                raise ValueError('Mapa con cobertura nodal incompleta')
            arrays = {'swmm': a.vol_inundacion_m3.to_numpy(), 'xgboost': a.volume_routed_m3.to_numpy(), 'mlp': b.volume_routed_m3.to_numpy()}
            vmax = max(1.0, *(float(v.max()) for v in arrays.values()))
            for model, values in arrays.items():
                title = f"{LABELS[name]} · factor {factor:.1f} · {model.upper()}"
                draw_map(a, values, title, INP, OUTPUT / f'mapa_{name}_f{factor:.1f}_{model}.png', vmax)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, name in zip(axes, LABELS):
        a = frames['xgboost'].loc[frames['xgboost'].shape_id == name].groupby('factor_mult').agg(swmm=('vol_inundacion_m3','sum'), xgb=('volume_routed_m3','sum'))
        b = frames['mlp'].loc[frames['mlp'].shape_id == name].groupby('factor_mult').volume_routed_m3.sum()
        ax.plot(a.index, a.swmm, marker='o', color='#333333', label='SWMM')
        ax.plot(a.index, a.xgb, marker='o', color='#4386c6', label='XGBoost')
        ax.plot(b.index, b, marker='o', color='#e87525', label='MLP')
        ax.set(title=LABELS[name], xlabel='Factor multiplicador', ylabel='Volumen total (m³)')
        ax.grid(alpha=.22); ax.legend(frameon=False)
    fig.savefig(OUTPUT / 'volumen_total_por_factor.png', dpi=180); plt.close(fig)
    manifest = {'scores_sha256': sha(DATA / 'scores.json'), 'missing_maps': missing,
                'scale': 'común entre SWMM, MLP y XGBoost en cada escenario',
                'files': {p.name: sha(p) for p in sorted(OUTPUT.glob('*.png'))}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps({'figures': len(manifest['files']), 'missing_maps': missing}, ensure_ascii=False))


if __name__ == '__main__':
    main()
