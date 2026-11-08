"""Read-only, post-hoc comparison of held-out node features against training ranges."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/experiments/prepared_chico_sur_norte_development_20261104_v1/65b26490afac36bd'
DATA = ROOT / 'outputs/experiments/evaluacion_reserva_conjunta_20261108_v1'
DB = ROOT / 'outputs/experiments/joint_chico_sur_norte_20261104_v1/joint.sqlite3'
OUT = ROOT / 'outputs/experiments/diagnostico_bloque_reservado_20261108'
FEATURES = ['q_pico_nodo', 'q_pico_acum_escalado', 'volumen_local_integrado_m3',
            'volumen_tributario_integrado_m3', 'duracion_efectiva_tributario_h',
            'tiempo_sobre_capacidad_h', 'integral_exceso_capacidad_m3']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def seconds_above(t, q, threshold):
    total = 0.
    for t0, t1, q0, q1 in zip(t[:-1], t[1:], q[:-1], q[1:]):
        if q0 >= threshold and q1 >= threshold:
            total += t1 - t0
        elif (q0 >= threshold) != (q1 >= threshold):
            crossing = t0 + (threshold - q0) * (t1 - t0) / (q1 - q0)
            total += crossing - t0 if q0 >= threshold else t1 - crossing
    return float(total)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    score = json.loads((DATA / 'scores.json').read_text(encoding='utf-8'))
    status = json.loads((DATA / 'status.json').read_text(encoding='utf-8'))
    seal = json.loads((DATA / 'prediction_seal.json').read_text(encoding='utf-8'))
    if (sha(DATA / 'scores.json') != status['scores_sha256'] or sha(DATA / 'prediction_seal.json') != score['seal_sha256']
            or sha(DATA / 'truth.parquet') != score['truth_sha256'] or sha(DATA / 'holdout_inputs.parquet') != seal['inputs_sha256']
            or sha(DB) != seal['source_database_sha256']):
        raise ValueError('source seal mismatch')
    contract = json.loads((ROOT / 'outputs/experiments/tune_joint_20261104/xgboost/study_contract.json').read_text(encoding='utf-8'))
    for name in ('keys.parquet', 'features.parquet', 'manifest.json'):
        path = (PREP / name).resolve()
        if contract['input_contract']['files_sha256'].get(str(path)) != sha(path):
            raise ValueError('prepared source differs from frozen study')
    train = pd.concat([pd.read_parquet(PREP / 'keys.parquet')[['network_id', 'node_id']],
                       pd.read_parquet(PREP / 'features.parquet')[FEATURES]], axis=1)
    maxima = train.groupby(['network_id', 'node_id'], as_index=False)[FEATURES].max()
    held = pd.read_parquet(DATA / 'holdout_inputs.parquet')
    held = held[(held.shape_id == 'shape_quasi_uniform_block') & held.factor_mult.isin([2., 3., 4., 5.])].copy()
    merged = held.merge(maxima, on=['network_id', 'node_id'], suffixes=('', '_train_max'), validate='many_to_one')
    if len(held) != len(merged) or len(held) != 4 * (160 + 522):
        raise ValueError('matched node coverage mismatch')
    comparison = []
    for (network, factor), group in merged.groupby(['network_id', 'factor_mult'], sort=True):
        for feature in FEATURES:
            valid = group[feature].notna() & group[feature + '_train_max'].notna()
            exceeded = valid & (group[feature] > group[feature + '_train_max'] + 1e-9)
            comparison.append({'network_id': int(network), 'factor': float(factor), 'feature': feature,
                               'valid_nodes': int(valid.sum()), 'above_same_node_training_max': int(exceeded.sum()),
                               'pct_above': 100 * float(exceeded.sum()) / int(valid.sum()) if valid.any() else None})
    shape_path = ROOT / 'data/hydrograph_shapes/shape_quasi_uniform_block.csv'
    joint = json.loads((ROOT / 'outputs/experiments/joint_chico_sur_norte_20261104_v1/joint_source_manifest.json').read_text(encoding='utf-8'))
    b_source = joint['source_manifests']['chico_norte']
    if sha(Path(b_source['path'])) != b_source['sha256']:
        raise ValueError('shape anchor source changed')
    b_manifest = json.loads(Path(b_source['path']).read_text(encoding='utf-8'))
    if sha(shape_path) != b_manifest['protocol_anchor']['execution_context']['shape_sources']['shape_quasi_uniform_block']['sha256']:
        raise ValueError('shape differs from anchored source')
    series = pd.read_csv(shape_path)
    t, q = series.time_h.to_numpy(dtype=float), series.q_norm.to_numpy(dtype=float)
    con = sqlite3.connect(DB.resolve().as_uri() + '?mode=ro', uri=True)
    ids = sorted(map(int, held.run_id.unique()))
    qa = pd.read_sql_query('SELECT run_id,continuity_error_pct,drainage_state,eligible FROM run_hydraulic_quality WHERE run_id IN (' + ','.join('?' for _ in ids) + ')', con, params=ids)
    con.close()
    if len(qa) != len(ids) or not (qa.eligible == 1).all():
        raise ValueError('QA coverage mismatch')
    OUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(comparison).to_csv(OUT / 'comparacion_maximos_por_nodo.csv', index=False)
    qa.to_csv(OUT / 'qa_escenarios.csv', index=False)
    result = {'purpose': 'post-hoc descriptive; no retraining or scenario exclusion',
              'shape': 'shape_quasi_uniform_block', 'duration_hours': float(t[-1] - t[0]),
              'equivalent_full_peak_hours': float(np.trapezoid(q, t)),
              'hours_at_or_above_90pct_peak': seconds_above(t, q, .9),
              'method': 'held feature versus maximum for same node and network across eligible training runs',
              'limitations': ['feature range comparison is not an SWMM stability test',
                  'training maximum is not a physical threshold', 'provisional QA and initial flow uncertainty remain'],
              'comparisons': comparison, 'qa': qa.to_dict(orient='records'),
              'scores_sha256': sha(DATA / 'scores.json'), 'prepared_manifest_sha256': sha(PREP / 'manifest.json')}
    (OUT / 'diagnostico.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'duration_hours': result['duration_hours'], 'equivalent_full_peak_hours': result['equivalent_full_peak_hours'],
                      'hours_at_or_above_90pct_peak': result['hours_at_or_above_90pct_peak'],
                      'factor5': [x for x in comparison if x['factor'] == 5.]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
