import json
from types import SimpleNamespace

import pandas as pd
import pytest


def prepared_fixture():
    return SimpleNamespace(prep_id='fixture', keys=pd.DataFrame({'shape_id': list('abcdef')}))


def test_study_resumes_without_repeating_trials_and_rejects_changed_contract(tmp_path, monkeypatch):
    from swmm_resilience.ml.bench import persistent_tuning as module
    calls = []
    def objective(trial, prepared, family, folds, **kwargs):
        value = trial.suggest_float('x', 0, 1)
        calls.append((trial.number, value))
        return value
    monkeypatch.setattr(module, '_objective', objective)
    contract = {'prepared_sha256': 'a' * 64, 'partition_sha256': 'b' * 64}
    first = module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path, contract, budget=3, max_new_trials=1)
    assert first['complete'] == 1 and len(calls) == 1
    second = module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path, contract, budget=3)
    assert second['complete'] == 3 and [x[0] for x in calls] == [0, 1, 2]
    with pytest.raises(ValueError, match='contract'):
        module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path,
                                    {**contract, 'partition_sha256': 'c' * 64}, budget=3)


def test_concurrent_execution_is_rejected(tmp_path):
    from swmm_resilience.ml.bench.persistent_tuning import exclusive_runner
    with exclusive_runner(tmp_path):
        with pytest.raises(RuntimeError, match='runner'):
            with exclusive_runner(tmp_path):
                pass


def test_interrupted_trial_is_failed_and_counts_toward_budget(tmp_path, monkeypatch):
    import optuna
    from swmm_resilience.ml.bench import persistent_tuning as module
    monkeypatch.setattr(module, '_objective', lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path, {'source': 'fixture'}, budget=2)
    monkeypatch.setattr(module, '_objective', lambda *args, **kwargs: 1.0)
    result = module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path, {'source': 'fixture'}, budget=2)
    study = optuna.load_study(study_name='joint_mlp', storage='sqlite:///' + (tmp_path / 'study.sqlite3').resolve().as_posix())
    assert [trial.state.name for trial in study.trials] == ['FAIL', 'COMPLETE']
    assert result['attempts'] == 2 and result['complete'] == 1 and result['failed'] == 1


def test_resuming_with_same_history_matches_continuous_search(tmp_path, monkeypatch):
    from swmm_resilience.ml.bench import persistent_tuning as module
    outcomes = []
    def objective(trial, *args, **kwargs):
        value = trial.suggest_float('x', 0, 1)
        outcomes.append(value)
        return value
    monkeypatch.setattr(module, '_objective', objective)
    module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path / 'one', {}, budget=12)
    uninterrupted = outcomes[:]
    outcomes.clear()
    module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path / 'two', {}, budget=12, max_new_trials=11)
    module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path / 'two', {}, budget=12)
    assert outcomes == uninterrupted


def test_missing_study_database_is_not_silently_recreated(tmp_path, monkeypatch):
    from swmm_resilience.ml.bench import persistent_tuning as module
    monkeypatch.setattr(module, '_objective', lambda *args, **kwargs: 1.0)
    module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path, {}, budget=2, max_new_trials=1)
    missing = tmp_path / 'missing'
    missing.mkdir()
    (missing / 'study_contract.json').write_bytes((tmp_path / 'study_contract.json').read_bytes())
    with pytest.raises(ValueError, match='database'):
        module.run_persistent_study(prepared_fixture(), 'mlp', missing, {}, budget=2)


def test_input_change_during_trial_is_not_marked_complete(tmp_path, monkeypatch):
    import optuna
    from swmm_resilience.ml.bench import persistent_tuning as module
    checks = []
    def validate():
        checks.append(True)
        if len(checks) == 2:
            raise ValueError('inputs changed')
    monkeypatch.setattr(module, '_objective', lambda *args, **kwargs: 1.0)
    with pytest.raises(ValueError, match='inputs changed'):
        module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path, {}, budget=2,
                                    integrity_check=validate)
    study = optuna.load_study(study_name='joint_mlp', storage='sqlite:///' + (tmp_path / 'study.sqlite3').resolve().as_posix())
    assert study.trials[0].state.name == 'FAIL'


def test_joint_tuning_rejects_reserved_shapes_or_extra_runs():
    from scripts.tune_joint_checkpointed import validate_development_membership
    prepared = SimpleNamespace(keys=pd.DataFrame({'run_id': [1, 2], 'shape_id': ['train', 'train']}))
    validate_development_membership(prepared, [1, 2], {'reserved'})
    with pytest.raises(ValueError, match='reserved'):
        validate_development_membership(prepared, [1, 2], {'train'})
    with pytest.raises(ValueError, match='runs'):
        validate_development_membership(prepared, [1], {'reserved'})


def test_nonfinite_objective_cannot_become_complete(tmp_path, monkeypatch):
    import optuna
    from swmm_resilience.ml.bench import persistent_tuning as module
    monkeypatch.setattr(module, '_objective', lambda *args, **kwargs: float('nan'))
    with pytest.raises(ValueError, match='not finite'):
        module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path, {}, budget=1)
    study = optuna.load_study(study_name='joint_mlp', storage='sqlite:///' + (tmp_path / 'study.sqlite3').resolve().as_posix())
    assert study.trials[0].state.name == 'FAIL'


def test_real_mlp_search_space_survives_sqlite_resume(tmp_path, monkeypatch):
    from swmm_resilience.ml.bench import persistent_tuning as module
    from swmm_resilience.ml.bench.tune import SEARCH_SPACES, _sample_params
    def objective(trial, *args, **kwargs):
        _sample_params(trial, SEARCH_SPACES['mlp']['classifier'], prefix='mlp_classifier_')
        _sample_params(trial, SEARCH_SPACES['mlp']['regressor'], prefix='mlp_regressor_')
        return 1.0
    monkeypatch.setattr(module, '_objective', objective)
    module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path, {}, budget=12, max_new_trials=1)
    result = module.run_persistent_study(prepared_fixture(), 'mlp', tmp_path, {}, budget=12)
    assert result['complete'] == 12


def test_real_objective_records_multinetwork_metrics_with_persistent_trial(tmp_path, monkeypatch):
    import numpy as np
    import optuna
    from swmm_resilience.ml.bench import tune
    from swmm_resilience.ml.bench import persistent_tuning as module
    keys = pd.DataFrame([{'shape_id': str(shape), 'network_id': network}
                         for shape in range(6) for network in (1, 2)])
    prepared = SimpleNamespace(prep_id='fixture', keys=keys,
                               y_clf=pd.Series([1] * len(keys)), y_reg=pd.Series([2.0] * len(keys)))
    def predict(_prepared, _family, classifier, regressor, _train, test):
        assert isinstance(classifier['hidden_sizes'], (tuple, list))
        return np.ones(len(test)), np.ones(len(test)), np.ones(len(test))
    monkeypatch.setattr(tune, '_fit_predict_fold', predict)
    result = module.run_persistent_study(prepared, 'mlp', tmp_path, {}, budget=2)
    study = optuna.load_study(study_name='joint_mlp', storage='sqlite:///' + (tmp_path / 'study.sqlite3').resolve().as_posix())
    assert result['complete'] == 2
    assert all(trial.value == 1.0 and set(trial.user_attrs['inner_network_rmse']) == {'1', '2'}
               for trial in study.trials)
