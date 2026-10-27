import json

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
    assert folds["group_column"].unique().tolist() == ["factor_mult"]
    for _, chunk in folds.groupby("fold_id"):
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"]
        expected_groups = sorted(str(value) for value in groups_25x7.iloc[test_idx].unique())
        assert chunk["held_out_groups"].unique().tolist() == [
            json.dumps(expected_groups, ensure_ascii=False)
        ]


def test_leave_one_shape_out_isolates_each_shape_with_explicit_provenance():
    keys = pd.DataFrame(
        {
            "shape_id": ["a", "a", "b", "b"],
            "factor_mult": [1.0, 2.0, 1.0, 2.0],
        }
    )

    folds = build_folds(keys["shape_id"], "LeaveOneShapeOut")

    for _, chunk in folds.groupby("fold_id"):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"]
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"]
        train_shapes = set(keys.loc[train_idx, "shape_id"])
        test_shapes = set(keys.loc[test_idx, "shape_id"])

        assert train_shapes.isdisjoint(test_shapes)
        assert len(test_shapes) == 1
        assert set(keys.loc[test_idx, "factor_mult"]) == {1.0, 2.0}
        assert chunk["group_column"].unique().tolist() == ["shape_id"]
        assert chunk["held_out_groups"].unique().tolist() == [
            json.dumps(sorted(test_shapes), ensure_ascii=False)
        ]


def test_leave_one_network_out_holds_out_every_row_for_one_network():
    # Node/scenario/event identifiers intentionally repeat between networks.
    keys = pd.DataFrame(
        {
            "network_id": ["A", "A", "B", "B", "C", "C"],
            "run_id": [1, 1, 1, 1, 1, 1],
            "scenario_id": [1, 2, 1, 2, 1, 2],
            "node_id": ["N1", "N2"] * 3,
            "event_id": ["event-1"] * 6,
        }
    )

    folds = build_folds(keys["network_id"], "LeaveOneNetworkOut")

    assert folds["fold_id"].nunique() == n_folds_for(
        keys["network_id"], "LeaveOneNetworkOut"
    ) == 3
    tested = folds.loc[folds["split"] == "test", "sample_idx"]
    assert sorted(tested) == list(range(len(keys)))
    assert tested.duplicated().sum() == 0
    for _, chunk in folds.groupby("fold_id"):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"]
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"]
        train_networks = set(keys.loc[train_idx, "network_id"])
        test_networks = set(keys.loc[test_idx, "network_id"])

        assert train_networks.isdisjoint(test_networks)
        assert len(test_networks) == 1
        assert chunk["group_column"].unique().tolist() == ["network_id"]
        assert chunk["held_out_groups"].unique().tolist() == [
            json.dumps(sorted(test_networks), ensure_ascii=False)
        ]


def test_leave_one_network_out_requires_two_networks():
    groups = pd.Series(["A", "A"], name="network_id")

    with pytest.raises(ValueError, match="LeaveOneNetworkOut needs at least 2 networks"):
        build_folds(groups, "LeaveOneNetworkOut")

    with pytest.raises(ValueError, match="LeaveOneNetworkOut needs at least 2 networks"):
        n_folds_for(groups, "LeaveOneNetworkOut")


def test_leave_one_network_out_rejects_null_network_ids():
    groups = pd.Series(["A", "B", None], name="network_id")

    with pytest.raises(ValueError, match="LeaveOneNetworkOut does not allow null network_id"):
        build_folds(groups, "LeaveOneNetworkOut")


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
        assert chunk["group_column"].unique().tolist() == ["factor_mult"]
        expected_groups = sorted(str(value) for value in groups_25x7.iloc[test_idx].unique())
        assert chunk["held_out_groups"].unique().tolist() == [
            json.dumps(expected_groups, ensure_ascii=False)
        ]


def test_loso_folds_are_ordered_by_group_value(groups_25x7):
    folds = build_folds(groups_25x7, "LOSO")
    first_test = folds[(folds["fold_id"] == 0) & (folds["split"] == "test")]["sample_idx"]
    assert groups_25x7.iloc[first_test].unique().tolist() == [0.2]


def test_build_all_folds_concatenates_both_protocols(groups_25x7):
    keys = pd.DataFrame({"factor_mult": groups_25x7})
    folds = build_all_folds(keys, ("LOSO", "GroupKFold5"))
    assert set(folds["protocol"]) == {"LOSO", "GroupKFold5"}
    assert len(folds) == len(build_folds(groups_25x7, "LOSO")) + len(
        build_folds(groups_25x7, "GroupKFold5")
    )


def test_groupkfold5_rejects_fewer_than_five_groups():
    groups = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], name="factor_mult")
    with pytest.raises(ValueError, match="GroupKFold5 needs at least 5 groups"):
        build_folds(groups, "GroupKFold5")


def test_n_folds_for_rejects_groupkfold5_with_fewer_than_five_groups():
    groups = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], name="factor_mult")
    with pytest.raises(ValueError, match="GroupKFold5 needs at least 5 groups"):
        n_folds_for(groups, "GroupKFold5")


def test_unknown_protocol_is_rejected(groups_25x7):
    with pytest.raises(ValueError, match="Unknown protocol"):
        build_folds(groups_25x7, "unknown")


def test_protocol_rejects_groups_from_the_wrong_column():
    wrong_groups = pd.Series(["a", "a", "b", "b"], name="shape_id")

    with pytest.raises(ValueError, match="factor_mult"):
        build_folds(wrong_groups, "LOSO")
