import numpy as np
import pandas as pd
import pytest
import optuna

from swmm_resilience.ml.bench.tune import build_tuning_folds, SEARCH_SPACES, _sample_params


def _keys_six_shapes() -> pd.DataFrame:
    """6 formas x 5 factores, para que GroupKFold(5) tenga margen."""
    shapes = ["s0", "s1", "s2", "s3", "s4", "s5"]
    factors = [1.0, 2.0, 3.0, 4.0, 5.0]
    rows = [
        {"shape_id": shape, "factor_mult": factor}
        for shape in shapes
        for factor in factors
    ]
    return pd.DataFrame(rows)


def test_build_tuning_folds_has_the_declared_columns():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, n_splits=5)
    assert tuple(folds.columns) == ("fold_id", "split", "sample_idx")
    assert folds["fold_id"].nunique() == 5


def test_build_tuning_folds_partitions_without_shape_leakage():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, n_splits=5)
    for _, chunk in folds.groupby("fold_id"):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
        train_shapes = set(keys["shape_id"].iloc[train_idx])
        test_shapes = set(keys["shape_id"].iloc[test_idx])
        assert train_shapes.isdisjoint(test_shapes)
        assert len(train_idx) + len(test_idx) == len(keys)


def test_build_tuning_folds_covers_every_row_across_folds_as_test_exactly_once():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, n_splits=5)
    test_rows = folds.loc[folds["split"] == "test", "sample_idx"]
    assert sorted(test_rows.tolist()) == list(range(len(keys)))
    assert not test_rows.duplicated().any()


def test_build_tuning_folds_is_deterministic_given_the_same_seed():
    keys = _keys_six_shapes()
    first = build_tuning_folds(keys, n_splits=5, seed=7)
    second = build_tuning_folds(keys, n_splits=5, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_build_tuning_folds_rejects_too_few_shapes_for_n_splits():
    keys = pd.DataFrame({"shape_id": ["a", "b", "c"], "factor_mult": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="5"):
        build_tuning_folds(keys, n_splits=5)


@pytest.mark.parametrize("family_name", ["xgboost", "random_forest", "linear", "svm", "mlp"])
@pytest.mark.parametrize("task", ["classifier", "regressor"])
def test_sample_params_returns_only_declared_keys_within_range(family_name, task):
    space = SEARCH_SPACES[family_name][task]
    study = optuna.create_study()
    trial = study.ask()

    params = _sample_params(trial, space, prefix=f"{family_name}_{task}_")

    assert set(params) == set(space)
    for name, spec in space.items():
        kind = spec[0]
        if kind in ("int", "float", "float_log"):
            low, high = spec[1], spec[2]
            assert low <= params[name] <= high
        elif kind == "categorical":
            assert params[name] in spec[1]


def test_sample_params_does_not_collide_between_classifier_and_regressor_in_the_same_trial():
    """El prefijo evita que Optuna vea el mismo nombre de parametro dos veces
    en un trial con rangos distintos (p.ej. xgboost.max_depth en ambas tareas)."""
    study = optuna.create_study()
    trial = study.ask()

    clf_params = _sample_params(
        trial, SEARCH_SPACES["xgboost"]["classifier"], prefix="xgboost_classifier_"
    )
    reg_params = _sample_params(
        trial, SEARCH_SPACES["xgboost"]["regressor"], prefix="xgboost_regressor_"
    )

    assert set(clf_params) == set(SEARCH_SPACES["xgboost"]["classifier"])
    assert set(reg_params) == set(SEARCH_SPACES["xgboost"]["regressor"])
