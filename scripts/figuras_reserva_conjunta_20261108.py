"""Generate fixed, separate flood maps and global parity plots from sealed scores."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from swmm_resilience.visualization._inp_parser import parse_coordinates, parse_conduits

DATA = ROOT / 'outputs/experiments/evaluacion_reserva_conjunta_20261108_v1'
OUTPUT = ROOT / 'docs/figures_reserva_conjunta_20261108_v2'
PREP = ROOT / 'outputs/experiments/prepared_chico_sur_norte_development_20261104_v1/65b26490afac36bd/manifest.json'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def draw_map(rows, node_values, title, inp, path, vmax):
    coords, pipes = parse_coordinates(inp), parse_conduits(inp)
    if not set(rows.node_id.astype(str)).issubset(coords):
        raise ValueError('map coordinates do not cover all nodes')
    fig, ax = plt.subplots(figsize=(13, 9), constrained_layout=True)
    for _, start, end in pipes:
        if start in coords and end in coords:
            ax.plot([coords[start][0], coords[end][0]], [coords[start][1], coords[end][1]],
                    color='#B0BEC5', linewidth=.7, zorder=1)
    wet = node_values > 0
    ids = rows.node_id.astype(str).to_numpy()
    xy = np.array([coords[n] for n in ids])
    norm = mcolors.PowerNorm(gamma=.5, vmin=0, vmax=max(vmax, 1.))
    points = ax.scatter(xy[wet, 0], xy[wet, 1], c=node_values[wet], cmap='YlOrRd', norm=norm,
                        s=30 + 220 * np.sqrt(node_values[wet] / max(vmax, 1.)), edgecolors='none', zorder=3)
    fig.colorbar(points, ax=ax, fraction=.035, pad=.02, label='Volumen de inundación por nodo (m³)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    ax.set_title(title, fontweight='bold', fontsize=15)
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)


def draw_parity(joined, family, path):
    actual = joined.vol_inundacion_m3.to_numpy(dtype=float)
    predicted = joined.volume_routed_m3.to_numpy(dtype=float)
    maxv = max(float(actual.max()), float(predicted.max()), 1.) * 1.05
    limit = np.log10(1 + maxv)
    fig, ax = plt.subplots(figsize=(7.5, 6.3), constrained_layout=True)
    ax.scatter(np.log10(1 + actual), np.log10(1 + predicted), s=10, alpha=.3,
               color='#ef8b2c', linewidths=0, rasterized=True)
    ax.plot([0, limit], [0, limit], '--', color='#59636E', linewidth=1.1, label='y = x')
    ticks = [x for x in [0, 1, 10, 100, 1000, 10000, 100000] if np.log10(1 + x) <= limit]
    ax.set_xticks(np.log10(1 + np.array(ticks)), [f'{x:,}' for x in ticks])
    ax.set_yticks(np.log10(1 + np.array(ticks)), [f'{x:,}' for x in ticks])
    ax.set_xlim(0, limit); ax.set_ylim(0, limit); ax.set_aspect('equal')
    ax.set_xlabel('Volumen SWMM (m³)'); ax.set_ylabel('Volumen predicho (m³)')
    ax.set_title('Paridad global · ' + family, loc='left', fontweight='bold')
    ax.grid(True, color='#CFD8E3', linewidth=.7, alpha=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    error = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    r2 = 1 - float(np.square(error).sum()) / denominator if denominator else float('nan')
    ax.text(.04, .96, f'RMSE = {np.sqrt(np.mean(error ** 2)):.2f} m³\n'
            f'MAE = {np.mean(np.abs(error)):.2f} m³\nR² = {r2:.3f}\nn = {len(actual):,}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox={'boxstyle': 'round,pad=0.35', 'fc': 'white', 'ec': '#BBBBBB', 'alpha': .94})
    fig.savefig(path, dpi=220, facecolor='white'); plt.close(fig)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    status = json.loads((DATA / 'status.json').read_text(encoding='utf-8'))
    scores_path = DATA / 'scores.json'
    if status['status'] != 'holdout_scored' or sha(scores_path) != status['scores_sha256']:
        raise ValueError('score seal mismatch')
    scores = json.loads(scores_path.read_text(encoding='utf-8'))
    seal_path = DATA / 'prediction_seal.json'
    if sha(seal_path) != scores['seal_sha256'] or sha(DATA / 'truth.parquet') != scores['truth_sha256']:
        raise ValueError('prediction or truth seal mismatch')
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    truth = pd.read_parquet(DATA / 'truth.parquet')
    prep = json.loads(PREP.read_text(encoding='utf-8'))
    inps = {int(n['network_id']): Path(n['inp_path']) for n in prep['source_networks']}
    for n in prep['source_networks']:
        if sha(inps[int(n['network_id'])]) != n['network_sha256']:
            raise ValueError('network INP hash mismatch')
    joined = {}
    for family in ('xgboost', 'mlp'):
        path = DATA / (family + '_predictions.parquet')
        if sha(path) != seal['models'][family]['predictions_sha256']:
            raise ValueError('prediction hash mismatch')
        joined[family] = pd.read_parquet(path).merge(truth, on=['network_id', 'run_id', 'node_id'], validate='one_to_one')
        if len(joined[family]) != len(truth):
            raise ValueError('truth coverage mismatch')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for family, label in [('xgboost', 'XGBoost'), ('mlp', 'MLP')]:
        draw_parity(joined[family], label, OUTPUT / ('paridad_global_' + family + '.png'))
    for network, run, name, factor in [(1, 55, 'Chico Sur', 1), (2, 1030, 'Chico Norte', 1),
                                       (1, 65, 'Chico Sur', 3), (2, 1040, 'Chico Norte', 3)]:
        x = joined['xgboost'].loc[joined['xgboost'].run_id == run].sort_values('node_id').reset_index(drop=True)
        m = joined['mlp'].loc[joined['mlp'].run_id == run].sort_values('node_id').reset_index(drop=True)
        if (x.empty or x.node_id.tolist() != m.node_id.tolist() or set(x.shape_id) != {'common_storm'}
                or set(x.factor_mult) != {float(factor)}):
            raise ValueError('fixed map scenario mismatch')
        actual = x.vol_inundacion_m3.to_numpy(dtype=float)
        xv = x.volume_routed_m3.to_numpy(); mv = m.volume_routed_m3.to_numpy()
        vmax = max(actual.max(), xv.max(), mv.max(), 1.)
        for label, values in [('SWMM', actual), ('XGBoost', xv), ('MLP', mv)]:
            draw_map(x, values, name + ' · ' + label + f' · factor {factor}', inps[network],
                     OUTPUT / f'mapa_{name.replace(" ", "_").lower()}_{label.lower()}_factor_{factor}.png', vmax)
    files = {p.name: sha(p) for p in sorted(OUTPUT.glob('*.png'))}
    (OUTPUT / 'manifest.json').write_text(json.dumps({'scores_sha256': sha(scores_path),
        'maps': 'common_storm factors 1.0 preregistered and 3.0 selected after scoring; same scale per network/scenario',
        'parity': 'global node-scenario pairs, log10(1+volume) display', 'files': files}, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'figures': len(files), 'output': str(OUTPUT)}))


if __name__ == '__main__':
    main()
