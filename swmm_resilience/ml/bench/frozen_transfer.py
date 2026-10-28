"""Two-stage, label-blind frozen transfer; no model fitting or selection."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import validate_volume_band_edges
from ...database.hydraulic_quality import build_hydraulic_quality_decision
from ...validation.network_admission import inspect_network_inp
from ...validation.network_policy import validate_network_profile
from ..contracts import resolve_feature_contract
from .evaluate import _nested_reporting_entry
from .train import load_candidate_pair, _sha256, MLP_SEEDS
from .models.mlp_family import SeedEnsembleClassifier, SeedEnsembleRegressor
from .schemas import KEY_COLUMNS


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode('utf-8')


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def register_transfer(candidate_dir, inp_path, domain_policy, b_runs, volume_bands_m3, output_root):
    """Lock candidate, test-network source and criteria before prediction.

    Run records must contain network_id, run_id, event_series_sha256 and a
    declared hydraulic policy. No B labels or fitted estimator is accepted.
    """
    candidate_dir = Path(candidate_dir).resolve()
    classifier, regressor, meta = load_candidate_pair(candidate_dir, candidate_dir)
    if meta.get('family') == 'mlp':
        expected_aggregation = {'probability': 'arithmetic_mean', 'conditional_volume': 'mean_after_expm1_m3', 'conditional_volume_min_m3_per_seed': 0.0, 'classification_threshold': 0.5, 'dry_routed_volume_m3': 0.0}
        members = meta.get('members') or []
        try:
            estimator_seeds = [[est.named_steps['model'].random_state for est in ensemble.estimators] for ensemble in (classifier, regressor)]
        except (AttributeError, KeyError, TypeError):
            estimator_seeds = []
        if (not isinstance(classifier, SeedEnsembleClassifier) or not isinstance(regressor, SeedEnsembleRegressor)
                or meta.get('random_seed') is not None or meta.get('random_seeds') != list(MLP_SEEDS)
                or meta.get('ensemble') != expected_aggregation
                or [member.get('random_seed') for member in members] != list(MLP_SEEDS)
                or any(not _digest(member.get('classifier_sha256')) or not _digest(member.get('regressor_sha256')) for member in members)
                or estimator_seeds != [list(MLP_SEEDS), list(MLP_SEEDS)]):
            raise ValueError('The approved five-seed MLP ensemble and aggregation policy are required')
    if meta.get('manifest_version') != 'paired_experiment_v1':
        raise ValueError('A candidate requires paired experiment provenance')
    if meta.get('objective') != 'macro_network_rmse' or meta.get('trial_metadata', {}).get('selection') != 'inner_tuning':
        raise ValueError('A-only model selection evidence and configured objective are required')
    networks = meta.get('source_networks') or []
    if len(networks) != 1 or not networks[0].get('runs'):
        raise ValueError('Candidate must identify exactly one A network and its runs')
    train_ids = sorted(set(r['run_id'] for r in networks[0]['runs']) - set(meta.get('excluded_run_ids', [])))
    evidence = meta['trial_metadata']
    study_path = Path(evidence.get('study_path', ''))
    if not study_path.is_file() or not _digest(evidence.get('study_sha256')) or _sha256(study_path) != evidence['study_sha256']:
        raise ValueError('Incomplete selection provenance: missing or altered selection study')
    study = _read(study_path)
    trials = study.get('trials', [])
    completed = [t for t in trials if isinstance(t, dict) and t.get('state') == 'COMPLETE' and isinstance(t.get('value'), (int, float)) and np.isfinite(t['value'])]
    if (not train_ids or not study.get('study_id') or study.get('prep_id') != meta['prep_id']
            or study.get('family') != meta['family'] or study.get('objective_metric') != meta['objective']
            or study.get('direction') != 'minimize' or not completed
            or study.get('best_classifier_params') != meta['classifier_params']
            or study.get('best_regressor_params') != meta['regressor_params']
            or study.get('best_value') != min(t['value'] for t in completed)
            or not all(t.get('objective_metric') == meta['objective'] and set((t.get('inner_network_rmse') or {}).keys()) == {str(networks[0]['network_id'])} and t.get('inner_fold_rmse') for t in completed)
            or not any(t.get('classifier_params') == study['best_classifier_params'] and t.get('regressor_params') == study['best_regressor_params'] and t['value'] == study['best_value'] for t in completed)):
        raise ValueError('Incomplete or inconsistent A-only selection provenance in selection study')
    if not b_runs or len({r['run_id'] for r in b_runs}) != len(b_runs):
        raise ValueError('B run IDs must be nonempty and unique')
    b_networks = {r['network_id'] for r in b_runs}
    if len(b_networks) != 1 or next(iter(b_networks)) == networks[0]['network_id']:
        raise ValueError('B must be a distinct single network')
    if any(not isinstance(r.get('event_series_components'), dict) or not _digest(r.get('event_series_sha256')) or _hash(r['event_series_components']) != r['event_series_sha256'] for r in b_runs):
        raise ValueError('Every B event must have a SHA-256 hash')
    contract = resolve_feature_contract(meta['feature_contract_id'])
    if domain_policy.get('feature_contract_version') != contract.contract_id:
        raise ValueError('B policy feature contract differs from A')
    bands = validate_volume_band_edges(volume_bands_m3, required=True)
    threshold = float(meta['target_threshold_m3'])
    if bands[0] < threshold:
        raise ValueError('Volume bands must start at or above flood threshold')
    inp_path = Path(inp_path).resolve()
    profile = inspect_network_inp(inp_path)
    issues = validate_network_profile(profile, domain_policy)
    # QA decisions are outputs; only their agreed policy is preregistered.
    records = []
    for run in b_runs:
        policy = run.get('hydraulic_policy')
        policy_sha256 = run.get('hydraulic_policy_sha256')
        if set(run) != {'network_id', 'run_id', 'event_series_sha256', 'event_series_components', 'hydraulic_policy', 'hydraulic_policy_sha256', 'expected_node_ids'}:
            raise ValueError('Preregistration accepts planned inputs and policy only, no run results or labels')
        if not isinstance(policy, dict) or not _digest(policy_sha256) or _hash(policy) != policy_sha256:
            raise ValueError('A complete hashed hydraulic policy is required for each B run')
        nodes = run.get('expected_node_ids')
        if not isinstance(nodes, list) or not nodes or any(not isinstance(node, str) or not node.strip() for node in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError('Each planned run needs unique nonempty expected_node_ids for coverage')
        records.append({'expected_node_ids': nodes, 'network_id': run['network_id'], 'run_id': run['run_id'],
                        'event_series_sha256': run['event_series_sha256'], 'event_series_components': run['event_series_components'], 'hydraulic_policy': policy,
                        'hydraulic_policy_sha256': policy_sha256})
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + meta['prep_id'][:12]
    directory = Path(output_root) / identifier / 'A_to_B_frozen'
    directory.mkdir(parents=True, exist_ok=False)
    registration = {'usage': 'synthetic_only_unbound_sources', 'schema': 'frozen_transfer_v1', 'candidate_dir': str(candidate_dir),
        'candidate_metadata_sha256': _sha256(candidate_dir / 'metadata.json'),
        'classifier_sha256': meta['classifier_sha256'], 'regressor_sha256': meta['regressor_sha256'],
        'prep_id': meta['prep_id'], 'a_network_id': networks[0]['network_id'], 'a_run_ids': train_ids,
        'contract_id': contract.contract_id, 'contract_sha256': contract.descriptor_sha256,
        'selection_objective': meta['objective'], 'selection_evidence': meta['trial_metadata'], 'selection_study': study,
        'classifier_params': meta['classifier_params'], 'regressor_params': meta['regressor_params'],
        'seeds': meta['random_seeds'], 'flood_threshold_m3': threshold, 'classification_threshold': 0.5,
        'volume_bands_m3': list(bands), 'b_inp_path': str(inp_path), 'b_inp_sha256': profile['source']['sha256'],
        'b_profile': profile, 'domain_policy': domain_policy, 'domain_issues': issues, 'b_runs': records,
        'metrics': ['rmse_m3', 'mae_m3', 'volume_bias_m3', 'confusion_matrix', 'recall', 'pr_auc', 'false_negatives_by_volume_band']}
    _write(directory / 'registration.json', registration)
    _write(directory / 'registration_seal.json', {'sha256': _sha256(directory / 'registration.json')})
    return directory


def _registration(directory):
    directory = Path(directory)
    if _sha256(directory / 'registration.json') != _read(directory / 'registration_seal.json')['sha256']:
        raise ValueError('Registration hash mismatch')
    return _read(directory / 'registration.json')


def predict_transfer(directory, keys, features, b_runs):
    """Validate admission and write sealed predictions; this interface has no labels."""
    directory = Path(directory)
    if any((directory / name).exists() for name in ('prediction_seal.json', 'rejections.json', 'predictions.csv', 'inputs.csv')):
        raise FileExistsError('Transfer prediction already attempted')
    reg = _registration(directory)
    candidate_dir = Path(reg['candidate_dir'])
    if _sha256(candidate_dir / 'metadata.json') != reg['candidate_metadata_sha256']:
        raise ValueError('Candidate changed after registration')
    classifier, regressor, meta = load_candidate_pair(candidate_dir, candidate_dir)
    if _sha256(Path(reg['b_inp_path'])) != reg['b_inp_sha256']:
        raise ValueError('B INP changed after registration')
    contract = resolve_feature_contract(reg['contract_id'])
    if tuple(features.columns) != contract.feature_names or len(features) != len(keys):
        raise ValueError('B feature columns or row count mismatch')
    if tuple(keys.columns) != KEY_COLUMNS:
        raise ValueError('B key columns must match the declared key schema without labels')
    key_columns = ['network_id', 'run_id', 'node_id']
    if any(c not in keys for c in key_columns) or keys[key_columns].isna().any().any() or keys.duplicated(key_columns).any():
        raise ValueError('B sample keys must be complete and unique')
    expected = {r['run_id']: r for r in reg['b_runs']}
    if any(set(r) != {'network_id', 'run_id', 'event_series_sha256', 'hydraulic_quality'} for r in b_runs):
        raise ValueError('B run record must contain identity, event hash and hydraulic quality only')
    actual = {r['run_id']: r for r in b_runs}
    if len(actual) != len(b_runs) or set(actual) != set(expected) or not set(keys.run_id).issubset(expected):
        raise ValueError('B run IDs differ from preregistration or feature keys')
    rejections = []
    if reg['domain_issues']:
        rejections.append({'scope':'network', 'issues':reg['domain_issues']})
    statuses = []
    for run_id, registered in expected.items():
        run = actual[run_id]
        if any(run.get(k) != registered[k] for k in ('network_id','event_series_sha256')) or not (keys.loc[keys.run_id == run_id, 'network_id'] == registered['network_id']).all():
            raise ValueError('B run identity or event hash changed')
        qa = run.get('hydraulic_quality') or {}
        try:
            policy = registered['hydraulic_policy']
            checked = build_hydraulic_quality_decision(continuity_error_pct=qa.get('continuity_error_pct'),
                max_continuity_error_pct=policy['max_continuity_error_pct'], continuity_limit_rationale=policy['continuity_limit_rationale'],
                finite_coverage_ok=qa.get('finite_coverage_ok'), drainage_state=qa.get('drainage_state'),
                initial_condition=qa.get('initial_condition'), flooding_report_manifest=qa.get('flooding_report'))
            valid = qa.get('policy_sha256') == registered['hydraulic_policy_sha256'] and checked['policy_sha256'] == registered['hydraulic_policy_sha256'] and qa.get('eligible') is True and checked['eligible'] is True
        except (KeyError, ValueError, TypeError):
            valid = False
            checked = {'reason_codes':['missing_or_invalid_quality']}
        reasons = list(checked['reason_codes'])
        if not valid and not reasons:
            reasons = ['quality_decision_or_policy_mismatch']
        status = {'run_id':run_id, 'eligible':valid, 'reason_codes':reasons}
        statuses.append(status)
        if not valid:
            rejections.append(status)
    eligible_runs = {status['run_id'] for status in statuses if status['eligible']}
    if not reg['domain_issues'] and not eligible_runs.issubset(set(keys.run_id)):
        raise ValueError('Every eligible B run must have feature rows')
    for run_id in eligible_runs:
        rows = keys.loc[keys.run_id == run_id]
        nodes = rows['node_id'].astype(str)
        if len(nodes) != len(expected[run_id]['expected_node_ids']) or set(nodes) != set(expected[run_id]['expected_node_ids']):
            raise ValueError(f'Incomplete or unexpected node coverage for B run {run_id}')
        shapes = rows['shape_id']
        if shapes.isna().any() or any(not isinstance(value, str) or not value.strip() for value in shapes) or shapes.nunique() != 1:
            raise ValueError(f'Invalid or inconsistent shape_id for B run {run_id}')
    if rejections:
        _write(directory / 'rejections.json', {'run_status':statuses, 'rejections':rejections})
    if reg['domain_issues'] or not eligible_runs:
        return
    eligible_mask = keys['run_id'].isin(eligible_runs).to_numpy()
    keys = keys.iloc[eligible_mask].reset_index(drop=True)
    features = features.iloc[eligible_mask].reset_index(drop=True)
    contract.validate_frame(features)
    probability = np.asarray(classifier.predict_proba(features))[:, 1]
    conditional = np.maximum(np.expm1(np.asarray(regressor.predict(features), dtype=float)), 0.0)
    if not np.isfinite(probability).all() or not np.isfinite(conditional).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Predictions must be finite probabilities and volumes')
    predicted = (probability >= reg['classification_threshold']).astype(int)
    with (directory / 'inputs.csv').open('x', encoding='utf-8', newline='') as handle:
        pd.concat([keys, features], axis=1).to_csv(handle, index=False)
    out = keys.copy()
    out['y_prob_clf'], out['y_pred_clf'] = probability, predicted
    out['volume_conditional_m3'] = conditional
    out['volume_routed_m3'] = np.where(predicted == 1, conditional, 0.)
    with (directory / 'predictions.csv').open('x', encoding='utf-8', newline='') as handle:
        out.to_csv(handle, index=False)
    _write(directory / 'prediction_seal.json', {'predictions_sha256':_sha256(directory / 'predictions.csv'), 'inputs_sha256':_sha256(directory / 'inputs.csv'),
        'registration_sha256':_sha256(directory / 'registration.json'), 'run_status':statuses,
        'rejections_sha256': _sha256(directory / 'rejections.json') if rejections else None})


def score_transfer(directory, labels):
    """Fail closed until verified source binding is implemented."""
    raise NotImplementedError('Real scoring is blocked: verified B source binding is not implemented')


def _score_synthetic_transfer(directory, labels):
    """Exercise scoring mechanics in synthetic unit tests only; not a real result."""
    directory = Path(directory)
    if (directory / 'synthetic_scores.json').exists():
        raise FileExistsError('Transfer score already exists')
    reg = _registration(directory)
    seal = _read(directory / 'prediction_seal.json')
    if seal['registration_sha256'] != _sha256(directory / 'registration.json') or seal['predictions_sha256'] != _sha256(directory / 'predictions.csv'):
        raise ValueError('Prediction seal mismatch')
    if _sha256(directory / 'inputs.csv') != seal['inputs_sha256']:
        raise ValueError('Inference input seal mismatch')
    if seal['rejections_sha256'] is not None and _sha256(directory / 'rejections.json') != seal['rejections_sha256']:
        raise ValueError('Rejection report seal mismatch')
    preds = pd.read_csv(directory / 'predictions.csv', dtype={'node_id':str}, keep_default_na=False)
    labels = labels.copy()
    labels['node_id'] = labels['node_id'].astype(str)
    keys = ['network_id','run_id','node_id']
    if labels[keys].isna().any().any() or labels.duplicated(keys).any():
        raise ValueError('B labels have incomplete or duplicate keys')
    joined = preds.merge(labels, on=keys, how='outer', validate='one_to_one', indicator=True)
    if not (joined['_merge'] == 'both').all():
        raise ValueError('B label keys do not match sealed predictions')
    volumes = joined['vol_inundacion_m3'].to_numpy(dtype=float)
    classes = joined['inunda'].to_numpy(dtype=float)
    if not np.isfinite(volumes).all() or (volumes < 0).any() or not np.isin(classes, [0,1]).all() or not np.array_equal(classes, (volumes >= reg['flood_threshold_m3']).astype(int)):
        raise ValueError('B labels are invalid or disagree with the preregistered flood threshold')
    def metrics(frame):
        return _nested_reporting_entry(frame['inunda'], frame['y_pred_clf'], frame['y_prob_clf'],
            frame['vol_inundacion_m3'], frame['volume_routed_m3'], reg['volume_bands_m3'], reg['flood_threshold_m3'])
    output = {'usage':'synthetic_only_unbound_sources', 'prediction_sha256':seal['predictions_sha256'], 'labels_sha256':_hash(labels.sort_values(keys).to_dict(orient='records')),
              'overall':metrics(joined), 'by_event':[{'shape_id':str(shape), **metrics(group)} for shape, group in joined.groupby('shape_id')]}
    _write(directory / 'synthetic_scores.json', output)
