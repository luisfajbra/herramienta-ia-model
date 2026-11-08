"""Seal predictions first, then score the approved joint-network shape holdout."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.prepare_joint_dataset import development_run_ids, verify_partition_hash
from swmm_resilience.database.joint_materialization import verify_joint_materialization, _require_no_wal
from swmm_resilience.ml.bench.evaluate import _nested_reporting_entry
from swmm_resilience.ml.bench.schemas import KEY_COLUMNS
from swmm_resilience.ml.bench.train import load_candidate_pair
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21

PACKAGE = ROOT / 'outputs/experiments/joint_chico_sur_norte_20261104_v1'
PARTITION = ROOT / 'outputs/experiments/particion_chico_sur_norte_20261104.json'
PARTITION_SHA = 'b81fd31efa6a0b28b51e66e7bdfb6255d80de37ea25f0cb57714010bb2e892f4'
CANDIDATES = ROOT / 'outputs/experiments/final_joint_candidates_20261108_v1'
OUTPUT = ROOT / 'outputs/experiments/evaluacion_reserva_conjunta_20261108_v1'
KEYS = ['network_id', 'run_id', 'node_id']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def validate_holdout_keys(frame, held_runs, train_runs, shapes, nodes_by_network):
    if set(held_runs) & set(train_runs):
        raise ValueError('holdout overlaps training runs')
    if set(frame.run_id.unique()) != set(held_runs) or frame.duplicated(KEYS).any():
        raise ValueError('holdout run or node coverage is incomplete')
    if not set(frame.shape_id.unique()).issubset(shapes):
        raise ValueError('holdout contains an unreserved shape')
    for _, group in frame.groupby('run_id', sort=False):
        if group.network_id.nunique() != 1 or group.shape_id.nunique() != 1:
            raise ValueError('holdout run has inconsistent identity')
        network = int(group.network_id.iloc[0])
        if set(group.node_id.astype(str)) != nodes_by_network[network]:
            raise ValueError('holdout node coverage mismatch')


def route_predictions(probability, log_volume):
    probability, log_volume = np.asarray(probability, dtype=float), np.asarray(log_volume, dtype=float)
    if (not np.isfinite(probability).all() or not np.isfinite(log_volume).all()
            or (probability < 0).any() or (probability > 1).any() or probability.shape != log_volume.shape):
        raise ValueError('predictions must be finite, aligned and probabilities between zero and one')
    with np.errstate(over='ignore', invalid='ignore'):
        conditional = np.maximum(np.expm1(log_volume), 0.)
    if not np.isfinite(conditional).all():
        raise ValueError('predicted volumes must be finite')
    predicted = (probability >= .5).astype(int)
    return predicted, np.where(predicted == 1, conditional, 0.)


def join_truth(pred, truth):
    if pred.duplicated(KEYS).any() or truth.duplicated(KEYS).any():
        raise ValueError('duplicate prediction or truth keys')
    joined = pred.merge(truth, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not (joined['_merge'] == 'both').all():
        raise ValueError('prediction and truth coverage differ')
    volume = joined.vol_inundacion_m3.to_numpy(dtype=float)
    classes = joined.inunda.to_numpy(dtype=float)
    if (not np.isfinite(volume).all() or (volume < 0).any()
            or not np.array_equal(classes, (volume >= 1.).astype(int))):
        raise ValueError('invalid truth or flood threshold mismatch')
    return joined.drop(columns=['_merge'])


def metric(frame):
    return _nested_reporting_entry(frame.inunda, frame.y_pred_clf, frame.y_prob_clf,
                                   frame.vol_inundacion_m3, frame.volume_routed_m3, (24., 108., 344.), 1.)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    verify_partition_hash(PARTITION, PARTITION_SHA)
    verified = verify_joint_materialization(PACKAGE)
    train_ids = development_run_ids(PACKAGE, PARTITION)
    part = json.loads(PARTITION.read_text(encoding='utf-8'))
    held_ids = part['run_ids']['holdout']
    if len(held_ids) != 321 or len(train_ids) != 1332:
        raise ValueError('unexpected approved partition sizes')
    source_hash = sha(PACKAGE / 'joint.sqlite3')
    candidates_status = json.loads((CANDIDATES / 'status.json').read_text(encoding='utf-8'))
    if candidates_status.get('stage') != 'both_candidates_fitted_not_holdout_evaluated':
        raise ValueError('final candidates are incomplete')
    con = sqlite3.connect((PACKAGE / 'joint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
    marks = ','.join('?' for _ in held_ids)
    columns = ','.join([*KEY_COLUMNS, *FEATURE_COLUMNS_V21])
    frame = pd.read_sql_query(f'SELECT {columns} FROM training_samples_v4_21 WHERE run_id IN ({marks}) ORDER BY network_id,run_id,node_id', con, params=held_ids)
    nodes = {int(n): set(group.node_id.astype(str)) for n, group in
             pd.read_sql_query('SELECT network_id,node_id FROM nodes', con).groupby('network_id')}
    validate_holdout_keys(frame, held_ids, train_ids, part['reserved_shapes'], nodes)
    X = TABULAR_V4_21.validate_frame(frame.loc[:, list(FEATURE_COLUMNS_V21)])
    OUTPUT.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(OUTPUT / 'holdout_inputs.parquet', index=False)
    models = {}
    for family, sub in [('xgboost', 'xgboost'), ('mlp', 'mlp/ensemble')]:
        directory = CANDIDATES / sub
        classifier, regressor, meta = load_candidate_pair(directory, directory)
        if (meta['family'] != family or meta['prep_id'] != '65b26490afac36bd'
                or set(meta['source_manifest']['run_ids']) != set(train_ids)
                or meta['feature_contract_id'] != 'tabular_v4_21'):
            raise ValueError('candidate training provenance mismatch')
        if family == 'mlp' and meta.get('random_seeds') != [42, 43, 44, 45, 46]:
            raise ValueError('MLP seeds mismatch')
        probability = np.asarray(classifier.predict_proba(X))[:, 1]
        classification, volumes = route_predictions(probability, regressor.predict(X))
        predictions = frame.loc[:, list(KEY_COLUMNS)].copy()
        predictions['y_prob_clf'], predictions['y_pred_clf'], predictions['volume_routed_m3'] = probability, classification, volumes
        path = OUTPUT / (family + '_predictions.parquet')
        predictions.to_parquet(path, index=False)
        models[family] = {'predictions_sha256': sha(path), 'classifier_sha256': meta['classifier_sha256'],
                          'regressor_sha256': meta['regressor_sha256'], 'metadata_sha256': sha(directory / 'metadata.json')}
        print(json.dumps({'predictions_sealed': family, 'rows': len(predictions)}), flush=True)
    seal = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_database_sha256': source_hash,
            'partition_sha256': PARTITION_SHA, 'inputs_sha256': sha(OUTPUT / 'holdout_inputs.parquet'),
            'holdout_run_ids': held_ids, 'reserved_shapes': part['reserved_shapes'], 'models': models,
            'limitations': ['known networks', 'previously explored project data', 'provisional hydraulic QA'],
            'joint_verification': verified}
    write_json(OUTPUT / 'prediction_seal.json', seal)
    # Only after both prediction tables are written and sealed do we fetch target labels.
    truth = pd.read_sql_query(f'''SELECT r.network_id,r.run_id,n.node_id,r.inunda,r.vol_inundacion_m3
        FROM node_results r JOIN nodes n ON n.node_pk=r.node_pk AND n.network_id=r.network_id
        WHERE r.run_id IN ({marks})''', con, params=held_ids)
    con.close()
    _require_no_wal(PACKAGE / 'joint.sqlite3', 'joint holdout source')
    if sha(PACKAGE / 'joint.sqlite3') != source_hash or sha(PARTITION) != PARTITION_SHA:
        raise ValueError('source changed after predictions')
    truth.to_parquet(OUTPUT / 'truth.parquet', index=False)
    results = {'schema': 'joint_shape_holdout_v1', 'seal_sha256': sha(OUTPUT / 'prediction_seal.json'),
               'truth_sha256': sha(OUTPUT / 'truth.parquet'), 'models': {}}
    for family in ('xgboost', 'mlp'):
        path = OUTPUT / (family + '_predictions.parquet')
        if sha(path) != models[family]['predictions_sha256']:
            raise ValueError('sealed predictions changed')
        joined = join_truth(pd.read_parquet(path), truth)
        report = {'overall_pooled': metric(joined), 'by_network': {}, 'by_shape_network': [], 'by_scenario': []}
        for network, group in joined.groupby('network_id', sort=True):
            report['by_network'][str(network)] = metric(group)
        report['macro_network_rmse_m3'] = float(np.mean([v['rmse_m3'] for v in report['by_network'].values()]))
        for (network, shape), group in joined.groupby(['network_id', 'shape_id'], sort=True):
            report['by_shape_network'].append({'network_id': int(network), 'shape_id': shape, **metric(group)})
        for (network, run), group in joined.groupby(['network_id', 'run_id'], sort=True):
            report['by_scenario'].append({'network_id': int(network), 'run_id': int(run),
                'shape_id': str(group.shape_id.iloc[0]), 'factor_mult': float(group.factor_mult.iloc[0]), **metric(group)})
        results['models'][family] = report
        print(json.dumps({'scored': family, 'macro_network_rmse_m3': report['macro_network_rmse_m3']}), flush=True)
    write_json(OUTPUT / 'scores.json', results)
    write_json(OUTPUT / 'status.json', {'status': 'holdout_scored', 'scores_sha256': sha(OUTPUT / 'scores.json')})


if __name__ == '__main__':
    main()
