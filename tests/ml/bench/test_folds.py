import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from swmm_resilience.ml.bench.folds import build_all_folds, build_folds, n_folds_for
from swmm_resilience.ml.bench.schemas import FOLD_COLUMNS


@pytest.fixture
def groups_25x7() -> pd.Series:
    """25 factores x 7 formas x 4 nodos, agrupado por factor."""
    factors = [round(0.2 * i, 1) for i in range(1, 26)]
    values = []
    for factor in factors:
        values.extend([factor] * 28)
    return pd.Series(values, name="factor_mult")


def test_loso_makes_one_fold_per_group(groups_25x7):
    folds = build_folds(groups_25x7, "LOSO")

    assert tuple(folds.columns) == FOLD_COLUMNS
    assert folds["fold_id"].nunique() == groups_25x7.nunique() == 25
    assert n_folds_for(groups_25x7, "LOSO") == 25


def test_every_sample_is_tested_exactly_once_per_protocol(groups_25x7):
    for protocol in ("LOSO", "GroupKFold5"):
        folds = build_folds(groups_25x7, protocol)
        tested = folds.loc[folds["split"] == "test", "sample_idx"]
        assert sorted(tested) == list(range(len(groups_25x7)))
        assert tested.duplicated().sum() == 0


def test_a_group_never_appears_in_train_and_test_of_the_same_fold(groups_25x7):
    for protocol in ("LOSO", "GroupKFold5"):
        folds = build_folds(groups_25x7, protocol)
        for fold_id, chunk in folds.groupby("fold_id"):
            train_groups = set(groups_25x7.iloc[chunk.loc[chunk["split"] == "train", "sample_idx"]])
            test_groups = set(groups_25x7.iloc[chunk.loc[chunk["split"] == "test", "sample_idx"]])
            assert not (train_groups & test_groups), f"{protocol} fold {fold_id} filtra grupos"


def test_train_and_test_together_cover_the_dataset_in_every_fold(groups_25x7):
    folds = build_folds(groups_25x7, "GroupKFold5")
    for _, chunk in folds.groupby("fold_id"):
        assert sorted(chunk["sample_idx"]) == list(range(len(groups_25x7)))


def test_groupkfold5_matches_sklearn_split_exactly(groups_25x7):
    """La paridad con evaluator.py depende de reproducir GroupKFold literalmente."""
    folds = build_folds(groups_25x7, "GroupKFold5")
    x_dummy = np.zeros((len(groups_25x7), 1))

    expected = list(GroupKFold(n_splits=5).split(x_dummy, None, groups_25x7.values))
    for fold_id, (train_idx, test_idx) in enumerate(expected):
        chunk = folds[folds["fold_id"] == fold_id]
        got_train = chunk.loc[chunk["split"] == "train", "sample_idx"].tolist()
        got_test = chunk.loc[chunk["split"] == "test", "sample_idx"].tolist()
        assert got_train == sorted(train_idx.tolist())
        assert got_test == sorted(test_idx.tolist())


def test_loso_folds_are_ordered_by_group_value(groups_25x7):
    folds = build_folds(groups_25x7, "LOSO")
    first_test = folds[(folds["fold_id"] == 0) & (folds["split"] == "test")]["sample_idx"]
    assert groups_25x7.iloc[first_test].unique().tolist() == [0.2]


def test_build_all_folds_concatenates_both_protocols(groups_25x7):
    folds = build_all_folds(groups_25x7, ("LOSO", "GroupKFold5"))
    assert set(folds["protocol"]) == {"LOSO", "GroupKFold5"}
    assert len(folds) == len(build_folds(groups_25x7, "LOSO")) + len(
        build_folds(groups_25x7, "GroupKFold5")
    )


def test_groupkfold5_rejects_fewer_than_five_groups():
    groups = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], name="factor_mult")
    with pytest.raises(ValueError, match="GroupKFold5 necesita al menos 5 grupos"):
        build_folds(groups, "GroupKFold5")


def test_n_folds_for_rejects_groupkfold5_with_fewer_than_five_groups():
    groups = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], name="factor_mult")
    with pytest.raises(ValueError, match="GroupKFold5 necesita al menos 5 grupos"):
        n_folds_for(groups, "GroupKFold5")


def test_unknown_protocol_is_rejected(groups_25x7):
    with pytest.raises(ValueError, match="Protocolo desconocido"):
        build_folds(groups_25x7, "LeaveOneShapeOut")
