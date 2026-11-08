"""Puntúa escenarios nuevos de Chico Norte con candidatos conjuntos ya fijados."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.evaluate_joint_holdout_20261108 import metric, route_predictions
from scripts.evaluacion_conjunta_hidrogramas_nuevos import validate_alignment
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.label_free_b_staging import stage_frozen_b_label_free
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.database.run_source_artifacts import verify_run_source_artifacts
from swmm_resilience.extraction.labels import extract_labels
from swmm_resilience.ml.bench.train import load_candidate_pair
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21

RUN = ROOT / 'outputs/experiments/conjuntos_escenarios_nuevos_chico_norte_20261108'
CANDIDATES = ROOT / 'outputs/experiments/final_joint_candidates_20261108_v1'
OUTPUT = RUN / 'evaluacion_modelos_conjuntos'
KEYS = ['run_id', 'node_id']
REGISTRY_SHA256 = '36f0b33854cba487c59acce83b4fb7a1b131a73a5c8b5b1b13a529cbe88ce1d5'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_event_novelty(keys, training_shapes, training_factors):
    if set(keys.shape_id.astype(str)) & set(training_shapes):
        raise ValueError('Una forma aparece en entrenamiento')
    if any(np.isclose(float(f), float(t), rtol=0, atol=1e-9)
           for f in keys.factor_mult.unique() for t in training_factors):
        raise ValueError('Un factor aparece en entrenamiento')


def main():
    if OUTPUT.exists() or (RUN / 'joint_staging.sqlite3').exists() or (RUN / 'experiment_source_manifest.json').exists():
        raise FileExistsError('La evaluación o sus fuentes derivadas ya existen')
    anchor = RUN.parent / (RUN.name + '_protocol_anchor.sqlite3')
    if sha(RUN / 'candidatos_registrados.json') != REGISTRY_SHA256:
        raise ValueError('El registro de candidatos ha cambiado')
    recorded = json.loads((RUN / 'candidatos_registrados.json').read_text(encoding='utf8'))
    conn = connect_managed_database(RUN / 'joint_staging.sqlite3')
    try:
        apply_migrations(conn)
        staged = stage_frozen_b_label_free(conn, RUN, feature_contract_id='tabular_v4_21',
                                           protocol_anchor_db_path=anchor)
        keys = staged['keys'].copy()
        keys['node_id'] = keys.node_id.astype(str)
        X = TABULAR_V4_21.validate_frame(staged['features'].loc[:, list(FEATURE_COLUMNS_V21)])
        if len(keys) != len(X) or keys.duplicated(KEYS).any():
            raise ValueError('Claves o variables incompletas')
        eligible = {int(x['run_id']) for x in staged['manifest']['run_order'] if x['eligible']}
        if not eligible or set(keys.run_id.astype(int)) != eligible:
            raise ValueError('Las variables no corresponden a las corridas elegibles')
        OUTPUT.mkdir(exist_ok=False)
        keys.to_parquet(OUTPUT / 'keys.parquet', index=False)
        X.to_parquet(OUTPUT / 'inputs.parquet', index=False)
        models = {}
        for family, sub in [('xgboost', 'xgboost'), ('mlp', 'mlp/ensemble')]:
            directory = CANDIDATES / sub
            hashes = {name: sha(directory / name) for name in ('metadata.json', 'classifier.joblib', 'regressor.joblib')}
            if hashes != recorded['models'][family]:
                raise ValueError('Los candidatos cambiaron después de registrarse')
            clf, reg, meta = load_candidate_pair(directory, directory)
            if meta['family'] != family or meta['feature_contract_id'] != 'tabular_v4_21' or len(meta['source_networks']) != 2:
                raise ValueError('Se requieren los candidatos conjuntos correctos')
            runs = [r for n in meta['source_networks'] for r in n['runs']]
            verify_event_novelty(keys, {r['shape_id'] for r in runs}, {r['factor_mult'] for r in runs})
            p = np.asarray(clf.predict_proba(X))[:, 1]
            label, volume = route_predictions(p, reg.predict(X))
            pred = keys.copy()
            pred['y_prob_clf'], pred['y_pred_clf'], pred['volume_routed_m3'] = p, label, volume
            pred.to_parquet(OUTPUT / f'{family}_predictions.parquet', index=False)
            models[family] = {'candidate_hashes': hashes, 'predictions_sha256': sha(OUTPUT / f'{family}_predictions.parquet')}
        seal = {'status': 'predicciones_antes_de_leer_RPT', 'candidate_registry_sha256': sha(RUN / 'candidatos_registrados.json'),
                'source_manifest_sha256': sha(staged['manifest_path']), 'inputs_sha256': sha(OUTPUT / 'inputs.parquet'),
                'keys_sha256': sha(OUTPUT / 'keys.parquet'), 'models': models}
        (OUTPUT / 'prediction_seal.json').write_text(json.dumps(seal, indent=2) + '\n', encoding='utf8')
        truth_parts = []
        for entry in staged['manifest']['run_order']:
            if not entry['eligible']:
                continue
            run_id = int(entry['run_id'])
            artifact = verify_run_source_artifacts(conn, run_id)['artifacts']['swmm_rpt']
            if artifact['sha256'] != entry['artifacts']['swmm_rpt']['sha256'] or sha(artifact['resolved_path']) != artifact['sha256']:
                raise ValueError('El RPT no coincide con su fuente verificada')
            nodes = keys.loc[keys.run_id == run_id, 'node_id'].tolist()
            labels = extract_labels(Path(artifact['resolved_path']), nodes, threshold_m3=1.0)
            labels['node_id'], labels['run_id'] = labels.node_id.astype(str), run_id
            truth_parts.append(labels[KEYS + ['inunda', 'vol_inundacion_m3']])
        truth = pd.concat(truth_parts, ignore_index=True)
        validate_alignment(keys, truth)
        if conn.execute('SELECT COUNT(*) FROM node_results').fetchone()[0] != 0:
            raise ValueError('Las etiquetas se escribieron en la base temporal')
        truth.to_parquet(OUTPUT / 'truth.parquet', index=False)
        scores = {'status': 'evaluacion_completa', 'scope': 'Chico Norte conocido; eventos nuevos y factores interpolados',
                  'excluded': [x for x in staged['manifest']['run_order'] if not x['eligible']],
                  'prediction_seal_sha256': sha(OUTPUT / 'prediction_seal.json'), 'truth_sha256': sha(OUTPUT / 'truth.parquet'),
                  'models': {}}
        for family in models:
            p = pd.read_parquet(OUTPUT / f'{family}_predictions.parquet')
            if sha(OUTPUT / f'{family}_predictions.parquet') != models[family]['predictions_sha256']:
                raise ValueError('Las predicciones cambiaron después de sellarse')
            joined = truth.merge(p, on=KEYS, validate='one_to_one')
            scores['models'][family] = {'overall': metric(joined), 'by_scenario': []}
            for (shape, factor), group in joined.groupby(['shape_id', 'factor_mult'], sort=True):
                scores['models'][family]['by_scenario'].append({'shape_id': shape, 'factor_mult': float(factor), **metric(group)})
        (OUTPUT / 'scores.json').write_text(json.dumps(scores, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
        print(json.dumps({'eligible_scenarios': len(eligible), 'excluded_scenarios': len(scores['excluded']),
                          'models': {k: v['overall'] for k, v in scores['models'].items()}}, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == '__main__':
    main()
