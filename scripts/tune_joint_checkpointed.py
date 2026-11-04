"""Run the approved joint-network tuning only on the verified development split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_joint_dataset import development_run_ids, verify_partition_hash
from swmm_resilience.database.joint_materialization import verify_joint_materialization
from swmm_resilience.ml.bench.persistent_tuning import run_persistent_study
from swmm_resilience.ml.bench.schemas import PreparedDataset


def validate_development_membership(prepared, expected_run_ids, reserved_shapes):
    if set(map(int, prepared.keys.run_id.unique())) != set(map(int, expected_run_ids)):
        raise ValueError('prepared runs do not match the development partition')
    if set(prepared.keys.shape_id.unique()) & set(reserved_shapes):
        raise ValueError('prepared dataset contains a reserved shape')


def file_hashes(paths):
    result = {}
    for path in paths:
        path = Path(path).resolve(strict=True)
        with path.open('rb') as f:
            result[str(path)] = hashlib.file_digest(f, 'sha256').hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('package', 'partition', 'prepared', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--partition-sha256', required=True)
    parser.add_argument('--family', choices=('mlp', 'xgboost'), required=True)
    parser.add_argument('--max-new-trials', type=int)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.max_new_trials is not None and args.max_new_trials < 1:
        parser.error('--max-new-trials must be positive')
    source_roots = [args.package.resolve(strict=True), args.prepared.resolve(strict=True)]
    output = args.output.resolve()
    if any(output == root or root in output.parents for root in source_roots):
        raise ValueError('study output must be outside source datasets')
    verify_partition_hash(args.partition, args.partition_sha256)
    verify_joint_materialization(args.package)
    ids = development_run_ids(args.package, args.partition)
    partition_data = json.loads(args.partition.read_text(encoding='utf-8'))
    files = [args.prepared / name for name in ('keys.parquet', 'features.parquet', 'targets.parquet',
                                               'folds.parquet', 'manifest.json', 'quality_report.json')]
    files += [args.partition, args.package / 'joint_source_manifest.json', Path(__file__),
              ROOT / 'scripts/prepare_joint_dataset.py']
    files += sorted((ROOT / 'swmm_resilience').rglob('*.py'))
    hashes = file_hashes(files)
    prepared = PreparedDataset.load(args.prepared)
    source = json.loads((args.package / 'joint_source_manifest.json').read_text(encoding='utf-8'))
    if (prepared.manifest.get('source_capture_method') != 'sealed_joint_byte_snapshot_v1'
            or prepared.manifest.get('source_snapshot_sha256') != source['joint_database']['sha256']):
        raise ValueError('prepared dataset was not captured from the verified joint database')
    validate_development_membership(prepared, ids, partition_data['reserved_shapes'])
    if prepared.manifest['protocols'] != ['ShapeGroupKFold5']:
        raise ValueError('prepared dataset does not use the approved shape protocol')
    for fold_id, rows in prepared.folds.groupby('fold_id'):
        for split in ('train', 'test'):
            subset = prepared.keys.iloc[rows.loc[rows['split'] == split, 'sample_idx'].to_numpy()]
            if set(subset.network_id.unique()) != {1, 2}:
                raise ValueError(f'fold {fold_id} {split} does not contain both networks')
    def check_inputs():
        if file_hashes(files) != hashes:
            raise ValueError('study input changed during tuning')
    check_inputs()
    if args.validate_only:
        print(json.dumps({'status': 'validated_not_trained', 'prep_id': prepared.prep_id,
                          'development_runs': len(ids), 'rows': len(prepared.keys)}, indent=2))
        return
    fixed = ({'classifier': {'epochs': 200, 'batch_size': 1024},
              'regressor': {'epochs': 200, 'batch_size': 1024}} if args.family == 'mlp' else
             {'classifier': {'n_jobs': 1}, 'regressor': {'n_jobs': 1}})
    result = run_persistent_study(prepared, args.family, args.output, {'files_sha256': hashes},
                                  budget=50, max_new_trials=args.max_new_trials,
                                  fixed_params=fixed, integrity_check=check_inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
