import pandas as pd
import pytest

from swmm_resilience.ml.bench.schemas import (
    FOLD_COLUMNS,
    KEY_COLUMNS,
    OOF_COLUMNS,
    PROTOCOLS,
    PreparedDataset,
)
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


def _tiny_prepared() -> PreparedDataset:
    n = 6
    keys = pd.DataFrame(
        {
            "run_id": [1, 1, 1, 2, 2, 2],
            "network_id": [1] * n,
            "scenario_id": [1, 1, 1, 2, 2, 2],
            "scenario_key": ["base@1.0"] * 3 + ["base@2.0"] * 3,
            "scenario_kind": ["base"] * n,
            "node_id": ["N0", "N1", "N2"] * 2,
            "factor_mult": [1.0] * 3 + [2.0] * 3,
            "shape_id": ["base"] * n,
        }
    )
    X = pd.DataFrame(
        {col: [float(i) for i in range(n)] for col in FEATURE_COLUMNS_V17}
    )
    folds = pd.DataFrame(
        {
            "protocol": ["LOSO"] * n,
            "fold_id": [0] * n,
            "sample_idx": list(range(n)),
            "split": ["train"] * 3 + ["test"] * 3,
        }
    )
    return PreparedDataset(
        prep_id="abc123def456789a",
        keys=keys,
        X=X,
        y_clf=pd.Series([0, 0, 1, 1, 1, 0], name="inunda"),
        y_reg=pd.Series([0.0, 0.0, 5.0, 7.0, 9.0, 0.0], name="vol_inundacion_m3"),
        folds=folds,
        manifest={"prep_id": "abc123def456789a", "contract_id": "tabular_v3_17"},
        quality={"n_rows": n},
    )


def test_column_constants_match_the_contract():
    assert KEY_COLUMNS == (
        "run_id",
        "network_id",
        "scenario_id",
        "scenario_key",
        "scenario_kind",
        "node_id",
        "factor_mult",
        "shape_id",
    )
    assert FOLD_COLUMNS == ("protocol", "fold_id", "sample_idx", "split")
    assert OOF_COLUMNS == (
        "sample_idx",
        "protocol",
        "fold_id",
        "y_pred_clf",
        "y_prob_clf",
        "y_pred_reg",
    )
    assert PROTOCOLS == ("LOSO", "GroupKFold5")


def test_save_then_load_roundtrips_every_component(tmp_path):
    prepared = _tiny_prepared()
    directory = prepared.save(tmp_path)

    assert directory == tmp_path / prepared.prep_id
    reloaded = PreparedDataset.load(directory)

    assert reloaded.prep_id == prepared.prep_id
    pd.testing.assert_frame_equal(reloaded.keys, prepared.keys)
    pd.testing.assert_frame_equal(reloaded.X, prepared.X)
    pd.testing.assert_series_equal(reloaded.y_clf, prepared.y_clf)
    pd.testing.assert_series_equal(reloaded.y_reg, prepared.y_reg)
    pd.testing.assert_frame_equal(reloaded.folds, prepared.folds)
    assert reloaded.manifest == prepared.manifest
    assert reloaded.quality == prepared.quality


def test_load_rejects_a_directory_missing_a_component(tmp_path):
    prepared = _tiny_prepared()
    directory = prepared.save(tmp_path)
    (directory / "folds.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="folds.parquet"):
        PreparedDataset.load(directory)


def test_feature_columns_must_match_the_contract_order():
    prepared = _tiny_prepared()
    shuffled = prepared.X.loc[:, list(reversed(FEATURE_COLUMNS_V17))]

    with pytest.raises(ValueError, match="orden del contrato"):
        PreparedDataset(
            prep_id=prepared.prep_id,
            keys=prepared.keys,
            X=shuffled,
            y_clf=prepared.y_clf,
            y_reg=prepared.y_reg,
            folds=prepared.folds,
            manifest=prepared.manifest,
            quality=prepared.quality,
        )


def test_keys_columns_must_match_the_contract():
    prepared = _tiny_prepared()
    shuffled_keys = prepared.keys.loc[:, list(reversed(KEY_COLUMNS))]

    with pytest.raises(ValueError, match="keys debe tener"):
        PreparedDataset(
            prep_id=prepared.prep_id,
            keys=shuffled_keys,
            X=prepared.X,
            y_clf=prepared.y_clf,
            y_reg=prepared.y_reg,
            folds=prepared.folds,
            manifest=prepared.manifest,
            quality=prepared.quality,
        )


def test_folds_columns_must_match_the_contract():
    prepared = _tiny_prepared()
    shuffled_folds = prepared.folds.loc[:, list(reversed(FOLD_COLUMNS))]

    with pytest.raises(ValueError, match="folds debe tener"):
        PreparedDataset(
            prep_id=prepared.prep_id,
            keys=prepared.keys,
            X=prepared.X,
            y_clf=prepared.y_clf,
            y_reg=prepared.y_reg,
            folds=shuffled_folds,
            manifest=prepared.manifest,
            quality=prepared.quality,
        )


def test_y_clf_name_must_match_the_contract():
    prepared = _tiny_prepared()
    renamed = prepared.y_clf.rename("not_inunda")

    with pytest.raises(ValueError, match="inunda"):
        PreparedDataset(
            prep_id=prepared.prep_id,
            keys=prepared.keys,
            X=prepared.X,
            y_clf=renamed,
            y_reg=prepared.y_reg,
            folds=prepared.folds,
            manifest=prepared.manifest,
            quality=prepared.quality,
        )


def test_y_reg_name_must_match_the_contract():
    prepared = _tiny_prepared()
    renamed = prepared.y_reg.rename("not_vol_inundacion_m3")

    with pytest.raises(ValueError, match="vol_inundacion_m3"):
        PreparedDataset(
            prep_id=prepared.prep_id,
            keys=prepared.keys,
            X=prepared.X,
            y_clf=prepared.y_clf,
            y_reg=renamed,
            folds=prepared.folds,
            manifest=prepared.manifest,
            quality=prepared.quality,
        )


def test_non_contiguous_index_is_rejected():
    prepared = _tiny_prepared()
    # Drop one row from every row-aligned component so the surviving index
    # ([0, 1, 3, 4, 5]) is no longer a default, contiguous RangeIndex. If
    # this went undetected, folds.sample_idx would silently misalign with
    # the rows it is meant to address positionally.
    filtered_keys = prepared.keys.drop(index=2)
    filtered_X = prepared.X.drop(index=2)
    filtered_y_clf = prepared.y_clf.drop(index=2)
    filtered_y_reg = prepared.y_reg.drop(index=2)

    with pytest.raises(ValueError, match="RangeIndex"):
        PreparedDataset(
            prep_id=prepared.prep_id,
            keys=filtered_keys,
            X=filtered_X,
            y_clf=filtered_y_clf,
            y_reg=filtered_y_reg,
            folds=prepared.folds,
            manifest=prepared.manifest,
            quality=prepared.quality,
        )
