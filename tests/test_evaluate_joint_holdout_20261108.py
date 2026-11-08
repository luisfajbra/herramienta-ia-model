import numpy as np
import pandas as pd
import pytest


def test_holdout_must_exclude_every_training_run_and_only_reserved_shapes():
    from scripts.evaluate_joint_holdout_20261108 import validate_holdout_keys
    frame = pd.DataFrame({'run_id': [10, 10, 11], 'network_id': [1, 1, 2],
                          'node_id': ['a', 'b', 'a'], 'shape_id': ['held', 'held', 'held']})
    validate_holdout_keys(frame, [10, 11], [1, 2], ['held'], {1: {'a', 'b'}, 2: {'a'}})
    with pytest.raises(ValueError, match='training'):
        validate_holdout_keys(frame, [10, 11], [10], ['held'], {1: {'a', 'b'}, 2: {'a'}})
    with pytest.raises(ValueError, match='coverage'):
        validate_holdout_keys(frame.iloc[:-1], [10, 11], [1], ['held'], {1: {'a', 'b'}, 2: {'a'}})


def test_cascade_and_join_reject_invalid_or_mismatched_data():
    from scripts.evaluate_joint_holdout_20261108 import route_predictions, join_truth
    cls, vol = route_predictions(np.array([.49, .5]), np.log1p([10., 20.]))
    assert cls.tolist() == [0, 1] and np.allclose(vol, [0., 20.])
    with pytest.raises(ValueError, match='finite'):
        route_predictions(np.array([np.nan]), np.array([0.]))
    pred = pd.DataFrame({'network_id': [1], 'run_id': [1], 'node_id': ['a']})
    truth = pd.DataFrame({'network_id': [1], 'run_id': [1], 'node_id': ['b'], 'inunda': [0], 'vol_inundacion_m3': [0.]})
    with pytest.raises(ValueError, match='coverage'):
        join_truth(pred, truth)
