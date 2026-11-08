"""Plot total-volume curves for the post-hoc diagnostic shape, from sealed outputs."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/experiments/evaluacion_reserva_conjunta_20261108_v1'
OUTPUT = ROOT / 'docs/figures_reserva_diagnostico_20261108'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    scores_path = DATA / 'scores.json'
    status = json.loads((DATA / 'status.json').read_text(encoding='utf-8'))
    if sha(scores_path) != status['scores_sha256']:
        raise ValueError('score hash mismatch')
    scores = json.loads(scores_path.read_text(encoding='utf-8'))
    if sha(DATA / 'prediction_seal.json') != scores['seal_sha256'] or sha(DATA / 'truth.parquet') != scores['truth_sha256']:
        raise ValueError('seal or truth hash mismatch')
    seal = json.loads((DATA / 'prediction_seal.json').read_text(encoding='utf-8'))
    truth = pd.read_parquet(DATA / 'truth.parquet')
    grouped = {}
    for family in ('xgboost', 'mlp'):
        path = DATA / (family + '_predictions.parquet')
        if sha(path) != seal['models'][family]['predictions_sha256']:
            raise ValueError('prediction hash mismatch')
        joined = pd.read_parquet(path).merge(truth, on=['network_id', 'run_id', 'node_id'], validate='one_to_one')
        if len(joined) != len(truth):
            raise ValueError('truth coverage mismatch')
        rows = joined[joined.shape_id == 'shape_quasi_uniform_block']
        grouped[family] = rows.groupby(['network_id', 'run_id', 'factor_mult'], as_index=False).agg(
            swmm_m3=('vol_inundacion_m3', 'sum'), predicted_m3=('volume_routed_m3', 'sum'))
    data = grouped['xgboost'].rename(columns={'predicted_m3': 'xgboost_m3'}).merge(
        grouped['mlp'].rename(columns={'predicted_m3': 'mlp_m3'}),
        on=['network_id', 'run_id', 'factor_mult', 'swmm_m3'], validate='one_to_one')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    data.to_csv(OUTPUT / 'volumen_por_factor.csv', index=False, float_format='%.17g')
    for network, name in [(1, 'Chico Sur'), (2, 'Chico Norte')]:
        group = data[data.network_id == network].sort_values('factor_mult')
        if len(group) != 25:
            raise ValueError('unexpected diagnostic scenario coverage')
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        for column, label, color in [('swmm_m3', 'SWMM', '#394a5b'), ('mlp_m3', 'MLP', '#e87525'), ('xgboost_m3', 'XGBoost', '#4386c6')]:
            ax.plot(group.factor_mult, group[column], marker='o', markersize=3, linewidth=2, label=label, color=color)
        ax.set_xlabel('Factor del hidrograma'); ax.set_ylabel('Volumen total de inundación (m³)')
        ax.set_title(name + ' · volumen por factor', loc='left', fontweight='bold')
        ax.grid(True, color='#CFD8E3', linewidth=.7, alpha=.8); ax.legend(frameon=False)
        fig.savefig(OUTPUT / ('volumen_factor_' + name.replace(' ', '_').lower() + '.png'), dpi=200, facecolor='white')
        plt.close(fig)
    files = {p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()}
    (OUTPUT / 'manifest.json').write_text(json.dumps({'selection': 'post-hoc diagnostic after observing shape-level errors',
        'shape': 'shape_quasi_uniform_block', 'scores_sha256': sha(scores_path), 'files': files},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'files': len(files)}))


if __name__ == '__main__':
    main()
