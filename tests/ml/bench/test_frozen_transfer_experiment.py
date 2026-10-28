from copy import deepcopy
import json
import hashlib
from pathlib import Path
import pandas as pd
import pytest

from swmm_resilience.database.hydraulic_quality import build_hydraulic_quality_decision
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.train import train_candidate
from swmm_resilience.ml.bench.tune import TuneResult, save_tune_result
from tests.validation.test_network_admission import _FIXTURE, _CLI_POLICY


EVENT = {'hydrograph_sha256':'a'*64}
EVENT_HASH = hashlib.sha256(json.dumps(EVENT, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


@pytest.fixture
def experiment(sql_training_db, tmp_path):
    prepared = prepare_dataset(sql_training_db, protocols=('LOSO',), flood_threshold_m3=1.0,
                               output_dir=tmp_path / 'prepared')
    params = {'n_estimators': 3, 'max_depth': 2, 'learning_rate': 0.3}
    trial = {'state':'COMPLETE', 'value':1.0, 'classifier_params':params, 'regressor_params':params, 'objective_metric':'macro_network_rmse', 'inner_fold_rmse':[1.0], 'inner_network_rmse':{str(prepared.manifest['source_network_ids'][0]):1.0}}
    result = TuneResult(family='xgboost', prep_id=prepared.prep_id, study_id='test', inner_cv={'method':'LeaveOneShapeOut'}, objective_metric='macro_network_rmse', direction='minimize', n_trials=1, seed=42, optuna_version='test', started_at_utc='test', finished_at_utc='test', duration_s=1.0, trials=[trial], best_classifier_params=params, best_regressor_params=params, best_value=1.0)
    study = save_tune_result(result, tmp_path / 'studies')
    evidence = {'selection':'inner_tuning', 'study_path':str(study), 'study_sha256':hashlib.sha256(study.read_bytes()).hexdigest()}
    candidate = train_candidate(prepared, 'xgboost', params, params, tmp_path / 'models', objective='macro_network_rmse', trial_metadata=evidence)
    inp = tmp_path / 'b.inp'
    inp.write_text(_FIXTURE, encoding='utf-8')
    policy = deepcopy(_CLI_POLICY)
    policy['feature_contract_version'] = candidate.metadata['feature_contract_id']
    keys = prepared.keys.iloc[:2].copy().reset_index(drop=True)
    keys['network_id'] = 999
    keys['run_id'] = 9999
    keys['node_id'] = ['J1', 'J2']
    x = prepared.X.iloc[:2].copy().reset_index(drop=True)
    decision = build_hydraulic_quality_decision(continuity_error_pct=1, max_continuity_error_pct=5,
        continuity_limit_rationale='provisional', finite_coverage_ok=True, drainage_state='drained',
        initial_condition={'policy_id':'test', 'state_sha256':'a'*64},
        flooding_report_manifest={'parser':'test','parser_version':'1','report_sha256':'b'*64})
    return candidate, inp, policy, keys, x, decision


def plan(run):
    qa = run['hydraulic_quality']
    return {key:run[key] for key in ('network_id','run_id','event_series_sha256')} | {'event_series_components':EVENT, 'hydraulic_policy':qa['policy'], 'hydraulic_policy_sha256':qa['policy_sha256']}


def test_predictions_sealed_before_labels_and_immutable(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer, predict_transfer, score_transfer
    candidate, inp, policy, keys, x, decision = experiment
    run = {'network_id': 999, 'run_id': 9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    directory = register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)],
                                  [24, 108, 344], tmp_path / 'outputs')
    assert not (directory / 'predictions.csv').exists()
    predict_transfer(directory, keys, x, [run])
    seal = json.loads((directory / 'prediction_seal.json').read_text())
    before = (directory / 'predictions.csv').read_bytes()
    labels = keys[['network_id','run_id','node_id']].copy()
    labels['vol_inundacion_m3'] = [0., 20.]
    labels['inunda'] = [0, 1]
    score_transfer(directory, labels)
    assert (directory / 'predictions.csv').read_bytes() == before
    assert seal['predictions_sha256']
    assert json.loads((directory / 'scores.json').read_text())['overall']['n_samples'] == 2
    with pytest.raises(FileExistsError):
        predict_transfer(directory, keys, x, [run])
    with pytest.raises(FileExistsError):
        score_transfer(directory, labels.assign(vol_inundacion_m3=[0., 80.]))


def test_invalid_hydraulic_run_is_reported_without_score(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer, predict_transfer
    candidate, inp, policy, keys, x, decision = experiment
    decision = deepcopy(decision)
    decision['eligible'] = False
    run = {'network_id': 999, 'run_id': 9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    directory = register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)],
                                  [24, 108, 344], tmp_path / 'outputs')
    predict_transfer(directory, keys, x, [run])
    assert (directory / 'rejections.json').exists()
    assert not (directory / 'prediction_seal.json').exists()
    assert not (directory / 'scores.json').exists()



def test_tampering_with_predictions_or_event_hash_is_rejected(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer, predict_transfer, score_transfer
    candidate, inp, policy, keys, x, decision = experiment
    run = {'network_id': 999, 'run_id': 9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    directory = register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)], [24,108,344], tmp_path/'outputs')
    with pytest.raises(ValueError, match='event hash'):
        predict_transfer(directory, keys, x, [{**run, 'event_series_sha256':'d'*64}])
    predict_transfer(directory, keys, x, [run])
    with (directory/'predictions.csv').open('a', encoding='utf8') as f:
        f.write('tampered')
    labels = keys[['network_id','run_id','node_id']].copy()
    labels['vol_inundacion_m3'] = [0., 20.]
    labels['inunda'] = [0, 1]
    with pytest.raises(ValueError, match='seal mismatch'):
        score_transfer(directory, labels)


def test_insufficient_selection_provenance_is_rejected(experiment, tmp_path, sql_training_db):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer
    candidate, inp, policy, keys, x, decision = experiment
    run = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    # A pair without verified selection cannot be preregistered.
    prepared = prepare_dataset(sql_training_db, protocols=('LOSO',), flood_threshold_m3=1.0, output_dir=tmp_path/'prepared2')
    candidate = train_candidate(prepared, 'xgboost', {'n_estimators':3}, {'n_estimators':3}, tmp_path/'other', objective='macro_network_rmse', trial_metadata={'selection':'inner_tuning', 'study_id':'test', 'trials':[1]})
    with pytest.raises(ValueError, match='selection provenance'):
        register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)], [24,108,344], tmp_path/'outputs')


def test_prediction_rejects_labels_smuggled_in_keys(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer, predict_transfer
    candidate, inp, policy, keys, x, decision = experiment
    run = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    directory = register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)], [24,108,344], tmp_path/'outputs')
    with pytest.raises(ValueError, match='key columns'):
        predict_transfer(directory, keys.assign(inunda=[0,1]), x, [run])


def test_registration_requires_hashed_selection_study(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer
    candidate, inp, policy, keys, x, decision = experiment
    run = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    Path(candidate.metadata['trial_metadata']['study_path']).unlink()
    with pytest.raises(ValueError, match='selection study'):
        register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)], [24,108,344], tmp_path/'outputs')


def test_can_preregister_policy_before_hydraulic_outputs_exist(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer
    candidate, inp, policy, keys, x, decision = experiment
    planned = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH,
               'event_series_components':EVENT, 'hydraulic_policy':decision['policy'], 'hydraulic_policy_sha256':decision['policy_sha256']}
    directory = register_transfer(candidate.classifier_path.parent, inp, policy, [planned], [24,108,344], tmp_path/'outputs')
    assert (directory/'registration_seal.json').exists()


def test_event_digest_must_match_event_manifest(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer
    candidate, inp, policy, keys, x, decision = experiment
    run = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    planned = plan(run) | {'event_series_components': {'hydrograph_sha256':'b'*64}}
    with pytest.raises(ValueError, match='event'):
        register_transfer(candidate.classifier_path.parent, inp, policy, [planned], [24,108,344], tmp_path/'outputs')


def test_prediction_rejects_labels_smuggled_in_run_record(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer, predict_transfer
    candidate, inp, policy, keys, x, decision = experiment
    run = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    directory = register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)], [24,108,344], tmp_path/'outputs')
    with pytest.raises(ValueError, match='run record'):
        predict_transfer(directory, keys, x, [{**run, 'labels': [0, 1]}])


def test_mixed_runs_only_score_eligible_and_report_ineligible(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer, predict_transfer, score_transfer
    candidate, inp, policy, keys, x, decision = experiment
    keys.loc[1, 'run_id'] = 9998
    good = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    bad_decision = deepcopy(decision)
    bad_decision['eligible'] = False
    bad = {'network_id':999, 'run_id':9998, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':bad_decision}
    directory = register_transfer(candidate.classifier_path.parent, inp, policy, [plan(good), plan(bad)], [24,108,344], tmp_path/'outputs')
    predict_transfer(directory, keys, x, [good, bad])
    predictions = pd.read_csv(directory/'predictions.csv')
    assert set(predictions.run_id) == {9999}
    rejected = json.loads((directory/'rejections.json').read_text())
    assert rejected['rejections'][0]['run_id'] == 9998
    assert rejected['rejections'][0]['reason_codes']
    labels = keys.loc[keys.run_id == 9999, ['network_id','run_id','node_id']].copy()
    labels['vol_inundacion_m3'] = 0.0
    labels['inunda'] = 0
    score_transfer(directory, labels)
    assert json.loads((directory/'scores.json').read_text())['overall']['n_samples'] == 1


def test_study_using_b_network_is_rejected(experiment, tmp_path, sql_training_db):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer
    candidate, inp, policy, keys, x, decision = experiment
    study_path = Path(candidate.metadata['trial_metadata']['study_path'])
    study = json.loads(study_path.read_text(encoding='utf8'))
    study['trials'][0]['inner_network_rmse']['999'] = 2.0
    study_path.write_text(json.dumps(study), encoding='utf8')
    evidence = {'selection':'inner_tuning', 'study_path':str(study_path), 'study_sha256':hashlib.sha256(study_path.read_bytes()).hexdigest()}
    prepared = prepare_dataset(sql_training_db, protocols=('LOSO',), flood_threshold_m3=1.0, output_dir=tmp_path/'prepared_again')
    candidate = train_candidate(prepared, 'xgboost', candidate.metadata['classifier_params'], candidate.metadata['regressor_params'], tmp_path/'other_models', objective='macro_network_rmse', trial_metadata=evidence)
    run = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    with pytest.raises(ValueError, match='selection provenance'):
        register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)], [24,108,344], tmp_path/'outputs')


def test_changed_b_labels_do_not_change_pair_or_sealed_predictions(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer, predict_transfer, score_transfer
    candidate, inp, policy, keys, x, decision = experiment
    run = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    candidate_hash = hashlib.sha256(candidate.classifier_path.read_bytes()).hexdigest()
    directories = [register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)], [24,108,344], tmp_path/name) for name in ('first', 'second')]
    for directory in directories:
        predict_transfer(directory, keys, x, [run])
    assert (directories[0]/'predictions.csv').read_bytes() == (directories[1]/'predictions.csv').read_bytes()
    labels = keys[['network_id','run_id','node_id']].copy()
    labels['inunda'] = [0,1]
    labels['vol_inundacion_m3'] = [0.,20.]
    score_transfer(directories[0], labels)
    labels['vol_inundacion_m3'] = [0.,80.]
    score_transfer(directories[1], labels)
    assert hashlib.sha256(candidate.classifier_path.read_bytes()).hexdigest() == candidate_hash
    assert (directories[0]/'predictions.csv').read_bytes() == (directories[1]/'predictions.csv').read_bytes()
    assert json.loads((directories[0]/'scores.json').read_text())['overall']['rmse_m3'] != json.loads((directories[1]/'scores.json').read_text())['overall']['rmse_m3']


def test_failed_run_needs_no_feature_rows(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer, predict_transfer
    candidate, inp, policy, keys, x, decision = experiment
    good = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    bad_qa = deepcopy(decision)
    bad_qa['eligible'] = False
    bad = {'network_id':999, 'run_id':9998, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':bad_qa}
    directory = register_transfer(candidate.classifier_path.parent, inp, policy, [plan(good), plan(bad)], [24,108,344], tmp_path/'outputs')
    predict_transfer(directory, keys, x, [good, bad])
    assert set(pd.read_csv(directory/'predictions.csv').run_id) == {9999}
    assert json.loads((directory/'rejections.json').read_text())['rejections'][0]['run_id'] == 9998


def test_seal_covers_inference_inputs(experiment, tmp_path):
    from swmm_resilience.ml.bench.frozen_transfer import register_transfer, predict_transfer, score_transfer
    candidate, inp, policy, keys, x, decision = experiment
    run = {'network_id':999, 'run_id':9999, 'event_series_sha256':EVENT_HASH, 'hydraulic_quality':decision}
    directory = register_transfer(candidate.classifier_path.parent, inp, policy, [plan(run)], [24,108,344], tmp_path/'outputs')
    predict_transfer(directory, keys, x, [run])
    seal = json.loads((directory/'prediction_seal.json').read_text())
    assert seal['inputs_sha256'] == hashlib.sha256((directory/'inputs.csv').read_bytes()).hexdigest()
    with (directory/'inputs.csv').open('a', encoding='utf8') as f:
        f.write('tampered')
    labels = keys[['network_id','run_id','node_id']].copy()
    labels['inunda'] = [0,1]
    labels['vol_inundacion_m3'] = [0.,20.]
    with pytest.raises(ValueError, match='input seal'):
        score_transfer(directory, labels)
