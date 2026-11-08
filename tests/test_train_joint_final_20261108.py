from types import SimpleNamespace
import pytest


def test_best_trial_requires_complete_approved_study():
    from scripts.train_joint_final_20261108 import select_best_trial
    good = SimpleNamespace(number=23, value=58.89, state=SimpleNamespace(name='COMPLETE'),
                           user_attrs={'objective_metric': 'macro_network_rmse',
                                       'classifier_params': {'depth': 3},
                                       'regressor_params': {'depth': 8}})
    study = SimpleNamespace(trials=[good] * 50, best_trial=good)
    assert select_best_trial(study)['trial_number'] == 23
    study.trials = [good] * 49
    with pytest.raises(ValueError, match='50'):
        select_best_trial(study)
    study.trials = [good] * 50
    good.user_attrs['objective_metric'] = 'pooled_rmse'
    with pytest.raises(ValueError, match='objective'):
        select_best_trial(study)


def test_reserved_shapes_and_extra_runs_cannot_enter_final_training():
    import pandas as pd
    from scripts.train_joint_final_20261108 import check_membership
    prepared = SimpleNamespace(keys=pd.DataFrame({'run_id': [1, 2], 'shape_id': ['a', 'b']}))
    check_membership(prepared, [1, 2], ['held'])
    with pytest.raises(ValueError, match='reserved'):
        check_membership(prepared, [1, 2], ['a'])
    with pytest.raises(ValueError, match='runs'):
        check_membership(prepared, [1], ['held'])
