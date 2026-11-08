"""Fit final joint candidates from frozen studies; never evaluate held-out shapes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna
from scripts.prepare_joint_dataset import development_run_ids, verify_partition_hash
from swmm_resilience.database.joint_materialization import verify_joint_materialization
from swmm_resilience.ml.bench.schemas import PreparedDataset
from swmm_resilience.ml.bench.train import train_candidate, train_mlp_ensemble


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def select_best_trial(study):
    if len(study.trials) != 50 or any(t.state.name != 'COMPLETE' for t in study.trials):
        raise ValueError('study must contain exactly 50 complete trials')
    trial = study.best_trial
    a = trial.user_attrs
    if a.get('objective_metric') != 'macro_network_rmse' or not math.isfinite(float(trial.value)):
        raise ValueError('study objective must be finite macro_network_rmse')
    if not isinstance(a.get('classifier_params'), dict) or not isinstance(a.get('regressor_params'), dict):
        raise ValueError('best trial lacks selected parameters')
    return {'trial_number': trial.number, 'value': float(trial.value),
            'classifier_params': a['classifier_params'], 'regressor_params': a['regressor_params'],
            'inner_network_rmse': a.get('inner_network_rmse')}


def check_membership(prepared, run_ids, reserved_shapes):
    if set(map(int, prepared.keys.run_id.unique())) != set(map(int, run_ids)):
        raise ValueError('prepared runs differ from development membership')
    if set(prepared.keys.shape_id.unique()) & set(reserved_shapes):
        raise ValueError('prepared dataset contains reserved shapes')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('package', 'partition', 'prepared', 'studies', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--partition-sha256', required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for source in (args.package.resolve(), args.prepared.resolve(), args.studies.resolve()):
        if output == source or source in output.parents:
            raise ValueError('output must be outside frozen inputs')
    verify_partition_hash(args.partition, args.partition_sha256)
    verify_joint_materialization(args.package)
    ids = development_run_ids(args.package, args.partition)
    partition = json.loads(args.partition.read_text(encoding='utf-8'))
    contracts = {}
    for family in ('xgboost', 'mlp'):
        c = json.loads((args.studies / family / 'study_contract.json').read_text(encoding='utf-8'))
        if (c.get('family') != family or c.get('prep_id') != '65b26490afac36bd'
                or c.get('objective') != 'macro_network_rmse' or c.get('budget_attempts') != 50):
            raise ValueError('study contract is incompatible')
        for filename, expected in c['input_contract']['files_sha256'].items():
            if sha256(filename) != expected:
                raise ValueError(f'study source changed: {filename}')
        contracts[family] = c
    prepared = PreparedDataset.load(args.prepared)
    check_membership(prepared, ids, partition['reserved_shapes'])
    source = json.loads((args.package / 'joint_source_manifest.json').read_text(encoding='utf-8'))
    if (prepared.prep_id != '65b26490afac36bd' or prepared.manifest.get('protocols') != ['ShapeGroupKFold5']
            or prepared.manifest.get('source_snapshot_sha256') != source['joint_database']['sha256']):
        raise ValueError('prepared source or protocol mismatch')
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'studies'
    snapshots.mkdir()
    selections = {}
    for family in ('xgboost', 'mlp'):
        source_db = (args.studies / family / 'study.sqlite3').resolve()
        frozen_db = snapshots / (family + '.sqlite3')
        with sqlite3.connect(source_db.as_uri() + '?mode=ro', uri=True) as src, sqlite3.connect(frozen_db) as dst:
            src.backup(dst)
        study = optuna.load_study(study_name='joint_' + family, storage='sqlite:///' + frozen_db.as_posix())
        chosen = select_best_trial(study)
        chosen['study_snapshot_sha256'] = sha256(frozen_db)
        chosen['study_contract_sha256'] = sha256(args.studies / family / 'study_contract.json')
        selections[family] = chosen
    (output / 'selections.json').write_text(json.dumps(selections, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    def status(stage):
        (output / 'status.json').write_text(json.dumps({'stage': stage, 'utc': datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
        print(json.dumps({'stage': stage}), flush=True)
    try:
        status('training_xgboost')
        x = selections['xgboost']
        train_candidate(prepared, 'xgboost', x['classifier_params'], x['regressor_params'], output,
                        objective='macro_network_rmse', inner_protocols=('ShapeGroupKFold5',),
                        trial_metadata=x, immutable=True)
        status('xgboost_complete_training_mlp')
        m = selections['mlp']
        train_mlp_ensemble(prepared, m['classifier_params'], m['regressor_params'], output,
                           objective='macro_network_rmse', inner_protocols=('ShapeGroupKFold5',), trial_metadata=m)
        status('both_candidates_fitted_not_holdout_evaluated')
    except BaseException:
        status('failed_check_logs_before_retry')
        raise


if __name__ == '__main__':
    main()
