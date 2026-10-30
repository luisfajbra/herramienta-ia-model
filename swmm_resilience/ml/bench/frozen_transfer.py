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
    if len(b_networks) != 1:
        raise ValueError('B must be a single network')
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
    a_network_sha256 = networks[0].get('network_sha256')
    if not _digest(a_network_sha256):
        raise ValueError('Candidate source network must have a verified INP SHA-256')
    b_inp_sha256 = _sha256(inp_path)
    if b_inp_sha256 == a_network_sha256:
        raise ValueError('B INP must identify a different source network from A')
    profile = inspect_network_inp(inp_path)
    if profile['source']['sha256'] != b_inp_sha256:
        raise ValueError('B INP changed while its network profile was being registered')
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


def score_transfer(directory, labels=None, *, conn=None, staged=None):
    """Score sealed B predictions using labels re-extracted from verified run RPTs.

    Caller-provided label frames remain forbidden. The prediction/input seals,
    frozen B manifest, SQLite artifact links, and each RPT hash are checked
    before the first target is materialized in memory.
    """
    if labels is not None or conn is None or staged is None:
        raise NotImplementedError(
            "Scoring requires verified B source binding; caller-supplied labels are not accepted"
        )
    directory = Path(directory)
    output_path = directory / "verified_scores.json"
    if output_path.exists():
        raise FileExistsError("Verified transfer score already exists")
    reg = _registration(directory)
    seal = _read(directory / "prediction_seal.json")
    if (seal.get("registration_sha256") != _sha256(directory / "registration.json")
            or seal.get("predictions_sha256") != _sha256(directory / "predictions.csv")
            or seal.get("inputs_sha256") != _sha256(directory / "inputs.csv")):
        raise ValueError("Prediction or input seal mismatch")
    rejection_sha = seal.get("rejections_sha256")
    if rejection_sha is not None:
        if not (directory / "rejections.json").is_file() or rejection_sha != _sha256(directory / "rejections.json"):
            raise ValueError("Rejection report seal mismatch")
    candidate_dir = Path(reg["candidate_dir"]).resolve(strict=True)
    candidate_hashes = {
        "metadata": (candidate_dir / "metadata.json", reg["candidate_metadata_sha256"]),
        "classifier": (candidate_dir / "classifier.joblib", reg["classifier_sha256"]),
        "regressor": (candidate_dir / "regressor.joblib", reg["regressor_sha256"]),
    }
    for role, (path, expected_hash) in candidate_hashes.items():
        if _sha256(path) != expected_hash:
            raise ValueError(f"Candidate {role} changed after B predictions were sealed")

    # Re-authenticate the complete B source package and require label-free SQLite.
    from ...database.label_free_b_staging import _verify_staged_state
    from ...database.run_source_artifacts import verify_run_source_artifacts
    from ...extraction.labels import extract_labels

    source_manifest = _verify_staged_state(conn, staged)
    root = Path(staged["experiment_root"]).resolve(strict=True)
    source_manifest_path = Path(staged["manifest_path"]).resolve(strict=True)
    source_manifest_hash = _sha256(source_manifest_path)
    if not _digest(source_manifest.get("manifest_sha256")):
        raise ValueError("B source manifest has no canonical seal")
    b_inp_record = source_manifest["run_order"][0]["artifacts"]["source_network_inp"]
    b_inp_path = Path(b_inp_record["path"])
    if not b_inp_path.is_absolute():
        b_inp_path = root / b_inp_path
    if (b_inp_path.resolve(strict=True) != Path(reg["b_inp_path"]).resolve(strict=True)
            or b_inp_record["sha256"] != reg["b_inp_sha256"]
            or _sha256(b_inp_path) != reg["b_inp_sha256"]):
        raise ValueError("Registered B INP differs from the verified staged source")

    ordered = source_manifest["run_order"]
    registered = {int(run["run_id"]): run for run in reg.get("b_runs", [])}
    if len(registered) != len(reg.get("b_runs", [])) or set(registered) != {int(item["run_id"]) for item in ordered}:
        raise ValueError("Registered B run IDs differ from the verified staged manifest")
    statuses = seal.get("run_status")
    if not isinstance(statuses, list):
        raise ValueError("Prediction seal has no per-run admission decisions")
    status_by_id = {int(item["run_id"]): item for item in statuses if isinstance(item, dict) and "run_id" in item}
    if len(status_by_id) != len(statuses) or set(status_by_id) != set(registered):
        raise ValueError("Prediction admission records differ from registered B runs")
    expected_rows = []
    labels_rows = []
    label_sources = []
    pending_label_runs = []
    eligible_run_ids = set()
    excluded_runs = []
    for entry in ordered:
        run_id = int(entry["run_id"])
        run_reg = registered[run_id]
        status = status_by_id[run_id]
        eligible = entry.get("eligible") is True
        if (status.get("eligible") is not eligible
                or status.get("reason_codes", []) != entry.get("reason_codes", [])):
            raise ValueError(f"Prediction admission decision differs from staged QA for run {run_id}")
        if not eligible:
            excluded_runs.append({"run_id": run_id, "scenario_key": entry["scenario_key"],
                                  "reason_codes": entry.get("reason_codes", [])})
            continue
        eligible_run_ids.add(run_id)
        event = entry["artifacts"]["event_source"]
        expected_components = {"source_artifact": {
            "path": event["path"], "sha256": event["sha256"], "byte_size": event["byte_size"],
        }}
        if (run_reg.get("network_id") != staged["network_id"]
                or run_reg.get("event_series_components") != expected_components
                or run_reg.get("event_series_sha256") != _hash(expected_components)):
            raise ValueError(f"Registered B event source differs from staged run {run_id}")
        if not status.get("eligible"):
            raise ValueError(f"Sealed prediction unexpectedly includes ineligible run {run_id}")
        scenario = conn.execute(
            """SELECT r.network_id,r.scenario_id,s.scenario_key,s.scenario_kind,s.shape_id,s.factor_mult
               FROM runs r JOIN scenarios s ON s.scenario_id=r.scenario_id WHERE r.run_id=?""",
            (run_id,),
        ).fetchone()
        if scenario is None:
            raise ValueError(f"Verified B scenario is missing for run {run_id}")
        nodes = [str(row[0]) for row in conn.execute(
            "SELECT node_id FROM nodes WHERE network_id=? ORDER BY node_id", (staged["network_id"],)
        )]
        if nodes != run_reg.get("expected_node_ids"):
            raise ValueError(f"Registered node inventory changed for B run {run_id}")
        expected_rows.extend({
            "run_id": run_id, "network_id": int(scenario[0]), "scenario_id": int(scenario[1]),
            "scenario_key": str(scenario[2]), "scenario_kind": str(scenario[3]),
            "node_id": node, "factor_mult": float(scenario[5]), "shape_id": str(scenario[4]),
        } for node in nodes)
        pending_label_runs.append((entry, run_id, scenario, nodes))

    if not eligible_run_ids:
        raise ValueError("No B runs passed frozen hydraulic QA; there are no transfer scores")
    preds = pd.read_csv(directory / "predictions.csv", dtype={"node_id": str, "scenario_key": str,
                                                               "scenario_kind": str, "shape_id": str},
                        keep_default_na=False)
    inputs = pd.read_csv(directory / "inputs.csv", dtype={"node_id": str, "scenario_key": str,
                                                           "scenario_kind": str, "shape_id": str})
    required_prediction_columns = [*KEY_COLUMNS, "y_prob_clf", "y_pred_clf", "volume_conditional_m3", "volume_routed_m3"]
    if preds.columns.tolist() != required_prediction_columns:
        raise ValueError("Sealed prediction file has an unsupported schema")
    expected_keys = pd.DataFrame(expected_rows, columns=list(KEY_COLUMNS))
    contract = resolve_feature_contract(reg["contract_id"])
    if inputs.columns.tolist() != [*KEY_COLUMNS, *contract.feature_names]:
        raise ValueError("Sealed inference input file has an unsupported feature schema")
    contract.validate_frame(inputs.loc[:, list(contract.feature_names)])
    def key_records(frame):
        normalized = frame.loc[:, list(KEY_COLUMNS)].copy()
        normalized["run_id"] = normalized["run_id"].astype(int)
        normalized["network_id"] = normalized["network_id"].astype(int)
        normalized["scenario_id"] = normalized["scenario_id"].astype(int)
        normalized["factor_mult"] = normalized["factor_mult"].astype(float)
        for column in ("scenario_key", "scenario_kind", "node_id", "shape_id"):
            normalized[column] = normalized[column].astype(str)
        return normalized.sort_values(["run_id", "node_id"]).reset_index(drop=True)
    if not key_records(preds).equals(key_records(expected_keys)):
        raise ValueError("Sealed B prediction keys differ from the verified eligible run/node inventory")
    if not key_records(inputs).equals(key_records(expected_keys)):
        raise ValueError("Sealed inference keys differ from the verified eligible run/node inventory")
    feature_table = "node_features" if contract.contract_id == "tabular_v3_17" else "node_features_v4_21"
    selected = ",".join(f"f.{column}" for column in contract.feature_names)
    for run_id in sorted(eligible_run_ids):
        rows = conn.execute(
            f"""SELECT n.node_id,{selected} FROM {feature_table} f
                JOIN nodes n ON n.node_pk=f.node_pk WHERE f.run_id=? ORDER BY n.node_id""",
            (run_id,),
        ).fetchall()
        stored = pd.DataFrame(rows, columns=["node_id", *contract.feature_names])
        actual = inputs.loc[inputs["run_id"].astype(int) == run_id].sort_values("node_id").reset_index(drop=True)
        stored["node_id"] = stored["node_id"].astype(str)
        if len(stored) != len(actual) or stored["node_id"].tolist() != actual["node_id"].tolist():
            raise ValueError(f"Sealed inference feature rows differ from SQLite for B run {run_id}")
        stored_values = stored.loc[:, list(contract.feature_names)].to_numpy(dtype=float)
        actual_values = actual.loc[:, list(contract.feature_names)].to_numpy(dtype=float)
        if not np.allclose(stored_values, actual_values, rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f"Sealed inference feature values differ from SQLite for B run {run_id}")
    probability = preds["y_prob_clf"].to_numpy(dtype=float)
    predicted_class = preds["y_pred_clf"].to_numpy(dtype=float)
    conditional_volume = preds["volume_conditional_m3"].to_numpy(dtype=float)
    routed_volume = preds["volume_routed_m3"].to_numpy(dtype=float)
    threshold = float(reg["classification_threshold"])
    if (not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any()
            or not np.isin(predicted_class, [0, 1]).all()
            or not np.isfinite(conditional_volume).all() or (conditional_volume < 0).any()
            or not np.isfinite(routed_volume).all() or (routed_volume < 0).any()
            or not np.array_equal(predicted_class.astype(int), (probability >= threshold).astype(int))
            or not np.allclose(routed_volume, np.where(predicted_class == 1, conditional_volume, 0.0),
                               rtol=1e-12, atol=1e-12)):
        raise ValueError("Sealed predictions violate the registered cascade or contain invalid values")
    classifier, regressor, _ = load_candidate_pair(candidate_dir, candidate_dir)
    expected_probability = np.asarray(classifier.predict_proba(inputs.loc[:, list(contract.feature_names)]))[:, 1]
    expected_conditional = np.maximum(
        np.expm1(np.asarray(regressor.predict(inputs.loc[:, list(contract.feature_names)]), dtype=float)), 0.0
    )
    if (not np.allclose(probability, expected_probability, rtol=1e-9, atol=1e-9)
            or not np.allclose(conditional_volume, expected_conditional, rtol=1e-9, atol=1e-9)):
        raise ValueError("Sealed B predictions do not reproduce from the registered candidate and verified features")

    # Target extraction starts only after the complete label-free prediction
    # table, its keys/features and its registered model outputs all verify.
    for entry, run_id, scenario, nodes in pending_label_runs:
        linked = verify_run_source_artifacts(conn, run_id)["artifacts"]
        rpt = linked["swmm_rpt"]
        rpt_path = Path(rpt["resolved_path"]).resolve(strict=True)
        if _sha256(rpt_path) != rpt["sha256"] or rpt["sha256"] != entry["artifacts"]["swmm_rpt"]["sha256"]:
            raise ValueError(f"B RPT source hash changed for run {run_id}")
        qa = entry.get("hydraulic_quality", {})
        report_qa = qa.get("flooding_report", {})
        if report_qa.get("report_sha256") != rpt["sha256"]:
            raise ValueError(f"B QA flooding-report link differs from RPT for run {run_id}")
        labels = extract_labels(rpt_path, nodes, threshold_m3=float(reg["flood_threshold_m3"]))
        label_report = labels.attrs.get("flooding_report_manifest", {})
        if label_report.get("report_sha256") != rpt["sha256"]:
            raise ValueError(f"B labels did not come from the QA-validated RPT for run {run_id}")
        labels = labels.copy()
        labels["node_id"] = labels["node_id"].astype(str)
        if (labels["node_id"].duplicated().any()
                or set(labels["node_id"]) != set(nodes)):
            raise ValueError(f"B RPT labels have incomplete or duplicate node coverage for run {run_id}")
        labels["network_id"] = int(scenario[0])
        labels["run_id"] = run_id
        labels_rows.append(labels[["network_id", "run_id", "node_id", "inunda", "vol_inundacion_m3"]])
        label_sources.append({"run_id": run_id, "scenario_key": str(scenario[2]),
                              "swmm_rpt_sha256": rpt["sha256"],
                              "flooding_report_sha256": report_qa["report_sha256"],
                              "label_rows": len(labels)})
    predictions_sha = seal["predictions_sha256"]
    truth = pd.concat(labels_rows, ignore_index=True)
    keys = ["network_id", "run_id", "node_id"]
    if truth.duplicated(keys).any():
        raise ValueError("Re-extracted B labels have duplicate run/node keys")
    joined = preds.merge(truth, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("Verified B labels do not cover the sealed prediction keys exactly")
    joined = joined.drop(columns=["_merge"])
    volumes = joined["vol_inundacion_m3"].to_numpy(dtype=float)
    classes = joined["inunda"].to_numpy(dtype=float)
    if (not np.isfinite(volumes).all() or (volumes < 0).any()
            or not np.isin(classes, [0, 1]).all()
            or not np.array_equal(classes, (volumes >= float(reg["flood_threshold_m3"])).astype(int))):
        raise ValueError("B labels are invalid or disagree with the registered flood threshold")
    metric = lambda frame: _nested_reporting_entry(
        frame["inunda"], frame["y_pred_clf"], frame["y_prob_clf"],
        frame["vol_inundacion_m3"], frame["volume_routed_m3"],
        reg["volume_bands_m3"], reg["flood_threshold_m3"],
    )
    output = {
        "schema_version": "verified_frozen_transfer_score_v1",
        "usage": "verified_staged_sources",
        "registration_sha256": _sha256(directory / "registration.json"),
        "prediction_sha256": predictions_sha,
        "input_sha256": seal["inputs_sha256"],
        "b_source_manifest_path": source_manifest_path.name,
        "b_source_manifest_sha256": source_manifest_hash,
        "b_inp_sha256": reg["b_inp_sha256"],
        "flood_threshold_m3": float(reg["flood_threshold_m3"]),
        "volume_bands_m3": list(reg["volume_bands_m3"]),
        "eligible_run_ids": sorted(eligible_run_ids),
        "excluded_runs": excluded_runs,
        "label_sources": label_sources,
        "labels_sha256": _hash(truth.sort_values(keys).to_dict(orient="records")),
        "overall": metric(joined),
        "by_event": [
            {"shape_id": str(shape), **metric(group)}
            for shape, group in joined.groupby("shape_id", sort=True)
        ],
        "by_scenario": [
            {"scenario_key": str(scenario_key), **metric(group)}
            for scenario_key, group in joined.groupby("scenario_key", sort=True)
        ],
    }
    _write(output_path, output)
    return output_path


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
