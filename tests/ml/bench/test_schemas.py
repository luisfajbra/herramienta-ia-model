import json

import pandas as pd
import pytest

from swmm_resilience.ml.bench.schemas import (
    FOLD_COLUMNS,
    KEY_COLUMNS,
    OOF_COLUMNS,
    PREP_ID_DESCRIPTOR_FIELDS,
    PROTOCOLS,
    PROTOCOL_GROUP_COLUMNS,
    PreparedDataset,
    compute_prep_id,
    group_column_for,
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
        [
            {
                "protocol": "LOSO",
                "fold_id": fold_id,
                "sample_idx": sample_idx,
                "split": (
                    "test"
                    if (fold_id == 0 and sample_idx < 3)
                    or (fold_id == 1 and sample_idx >= 3)
                    else "train"
                ),
                "group_column": "factor_mult",
                "held_out_groups": f'["{fold_id + 1}.0"]',
            }
            for fold_id in (0, 1)
            for sample_idx in range(n)
        ],
        columns=list(FOLD_COLUMNS),
    )
    descriptor = {
        "contract_id": "tabular_v3_17",
        "feature_contract_sha256": "b" * 64,
        "run_ids": [1, 2],
        "n_rows": n,
        "feature_columns": list(FEATURE_COLUMNS_V17),
        "flood_threshold_m3": 1.0,
        "sort_key": ["run_id", "node_id"],
        "protocols": ["LOSO"],
        "protocol_definitions": {
            "LOSO": {"group_column": "factor_mult"}
        },
        "source_capture_method": "sqlite_backup_readonly_v1",
        "source_snapshot_sha256": "d" * 64,
        "training_frame_fingerprint_version": "training_frame_canonical_v1",
        "training_frame_sha256": "c" * 64,
    }
    prep_id = compute_prep_id(
        {field: descriptor[field] for field in PREP_ID_DESCRIPTOR_FIELDS}
    )
    return PreparedDataset(
        prep_id=prep_id,
        keys=keys,
        X=X,
        y_clf=pd.Series([0, 0, 1, 1, 1, 0], name="inunda"),
        y_reg=pd.Series([0.0, 0.0, 5.0, 7.0, 9.0, 0.0], name="vol_inundacion_m3"),
        folds=folds,
        manifest=dict(descriptor, prep_id=prep_id),
        quality={"n_rows": n},
    )


def _write_manifest(directory, manifest):
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def _rebuild_prepared(prepared, *, folds=None, manifest=None):
    return PreparedDataset(
        prep_id=prepared.prep_id,
        keys=prepared.keys,
        X=prepared.X,
        y_clf=prepared.y_clf,
        y_reg=prepared.y_reg,
        folds=prepared.folds if folds is None else folds,
        manifest=prepared.manifest if manifest is None else manifest,
        quality=prepared.quality,
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
    assert FOLD_COLUMNS == (
        "protocol",
        "fold_id",
        "sample_idx",
        "split",
        "group_column",
        "held_out_groups",
    )
    assert OOF_COLUMNS == (
        "sample_idx",
        "protocol",
        "fold_id",
        "y_pred_clf",
        "y_prob_clf",
        "y_pred_reg",
    )
    assert PROTOCOLS == ("LOSO", "GroupKFold5", "LeaveOneShapeOut")


def test_protocols_select_their_required_group_column():
    assert PROTOCOL_GROUP_COLUMNS == {
        "LOSO": "factor_mult",
        "GroupKFold5": "factor_mult",
        "LeaveOneShapeOut": "shape_id",
    }
    assert group_column_for("LOSO") == "factor_mult"
    assert group_column_for("GroupKFold5") == "factor_mult"
    assert group_column_for("LeaveOneShapeOut") == "shape_id"


def test_group_column_for_rejects_an_unknown_protocol():
    with pytest.raises(ValueError, match="Protocolo desconocido"):
        group_column_for("unknown")


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


@pytest.mark.parametrize(
    "source_hash",
    [None, "", "a" * 63, "g" * 64],
    ids=["missing", "empty", "wrong-length", "non-hex"],
)
def test_load_rejects_missing_or_invalid_source_snapshot_hash(tmp_path, source_hash):
    prepared = _tiny_prepared()
    directory = prepared.save(tmp_path)
    manifest = dict(prepared.manifest)
    if source_hash is None:
        manifest.pop("source_snapshot_sha256")
    else:
        manifest["source_snapshot_sha256"] = source_hash
    _write_manifest(directory, manifest)

    with pytest.raises(ValueError, match="source_snapshot_sha256.*--bench-prepare"):
        PreparedDataset.load(directory)


def test_load_rejects_legacy_manifest_with_only_source_database_hash(tmp_path):
    prepared = _tiny_prepared()
    directory = prepared.save(tmp_path)
    manifest = dict(prepared.manifest)
    for field in (
        "source_capture_method",
        "source_snapshot_sha256",
        "training_frame_fingerprint_version",
        "training_frame_sha256",
    ):
        manifest.pop(field, None)
    manifest["source_db_sha256"] = "a" * 64
    _write_manifest(directory, manifest)

    with pytest.raises(
        ValueError,
        match=r"(?:source_snapshot_sha256|identity).*--bench-prepare",
    ):
        PreparedDataset.load(directory)


@pytest.mark.parametrize(
    "field",
    [
        field
        for field in PREP_ID_DESCRIPTOR_FIELDS
        if field != "training_frame_sha256"
    ],
)
def test_load_rejects_a_manifest_missing_a_prep_descriptor_field(tmp_path, field):
    prepared = _tiny_prepared()
    directory = prepared.save(tmp_path)
    manifest = dict(prepared.manifest)
    manifest.pop(field)
    _write_manifest(directory, manifest)

    with pytest.raises(ValueError, match="descriptor.*--bench-prepare"):
        PreparedDataset.load(directory)


def test_load_rejects_a_forged_prep_id(tmp_path):
    prepared = _tiny_prepared()
    directory = prepared.save(tmp_path)
    manifest = dict(prepared.manifest, prep_id="0" * 16)
    _write_manifest(directory, manifest)

    with pytest.raises(ValueError, match="prep_id.*--bench-prepare"):
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


@pytest.mark.parametrize("column", ["group_column", "held_out_groups"])
def test_fold_provenance_must_not_be_null(column):
    prepared = _tiny_prepared()
    invalid_folds = prepared.folds.copy()
    invalid_folds.loc[0, column] = None

    with pytest.raises(ValueError, match="procedencia"):
        PreparedDataset(
            prep_id=prepared.prep_id,
            keys=prepared.keys,
            X=prepared.X,
            y_clf=prepared.y_clf,
            y_reg=prepared.y_reg,
            folds=invalid_folds,
            manifest=prepared.manifest,
            quality=prepared.quality,
        )


def test_fold_provenance_must_match_groups_reconstructed_from_test_rows():
    prepared = _tiny_prepared()
    invalid_folds = prepared.folds.copy()
    invalid_folds["held_out_groups"] = '["1.0"]'

    with pytest.raises(ValueError, match="held_out_groups.*--bench-prepare"):
        _rebuild_prepared(prepared, folds=invalid_folds)


@pytest.mark.parametrize(
    "manifest_protocols",
    [[], ["LOSO", "LeaveOneShapeOut"]],
    ids=["fold-protocol-extra", "fold-protocol-missing"],
)
def test_fold_protocols_must_equal_manifest_protocols(manifest_protocols):
    prepared = _tiny_prepared()
    manifest = dict(prepared.manifest, protocols=manifest_protocols)

    with pytest.raises(ValueError, match="protocolos.*folds.*--bench-prepare"):
        _rebuild_prepared(prepared, manifest=manifest)


@pytest.mark.parametrize("sample_idx", [None, 6], ids=["missing", "out-of-range"])
def test_fold_provenance_rejects_invalid_sample_indices(sample_idx):
    prepared = _tiny_prepared()
    invalid_folds = prepared.folds.copy()
    invalid_folds.loc[3, "sample_idx"] = sample_idx

    with pytest.raises(
        ValueError,
        match=r"(?:procedencia de folds|folds\.sample_idx).*--bench-prepare",
    ):
        _rebuild_prepared(prepared, folds=invalid_folds)


@pytest.mark.parametrize("mutation", ["omitted", "duplicated"])
def test_each_fold_must_partition_every_sample_exactly_once(mutation):
    prepared = _tiny_prepared()
    invalid_folds = prepared.folds.copy()
    fold_zero_rows = invalid_folds["fold_id"] == 0
    if mutation == "omitted":
        invalid_folds = invalid_folds.drop(
            invalid_folds[fold_zero_rows].index[0]
        )
    else:
        first_two = invalid_folds[fold_zero_rows].index[:2]
        invalid_folds.loc[first_two[0], "sample_idx"] = invalid_folds.loc[
            first_two[1], "sample_idx"
        ]

    with pytest.raises(ValueError, match="partición.*--bench-prepare"):
        _rebuild_prepared(prepared, folds=invalid_folds)


def test_a_held_out_group_must_not_leak_into_train():
    prepared = _tiny_prepared()
    invalid_folds = prepared.folds.copy()
    fold_zero = invalid_folds["fold_id"] == 0
    invalid_folds.loc[fold_zero & (invalid_folds["sample_idx"] == 0), "split"] = "train"
    invalid_folds.loc[fold_zero & (invalid_folds["sample_idx"] == 3), "split"] = "test"
    invalid_folds.loc[fold_zero, "held_out_groups"] = '["1.0", "2.0"]'

    with pytest.raises(
        ValueError,
        match=r"(?:grupos|held_out_groups|procedencia).*--bench-prepare",
    ):
        _rebuild_prepared(prepared, folds=invalid_folds)


def test_loso_rejects_more_than_one_held_out_group_per_fold():
    prepared = _tiny_prepared()
    invalid_folds = prepared.folds[prepared.folds["fold_id"] == 0].copy()
    invalid_folds["split"] = "test"
    invalid_folds["held_out_groups"] = '["1.0", "2.0"]'

    with pytest.raises(ValueError, match="LOSO.*grupo.*--bench-prepare"):
        _rebuild_prepared(prepared, folds=invalid_folds)


def test_groupkfold5_requires_five_canonical_fold_ids():
    prepared = _tiny_prepared()
    invalid_folds = prepared.folds.copy()
    invalid_folds["protocol"] = "GroupKFold5"
    manifest = dict(prepared.manifest, protocols=["GroupKFold5"])

    with pytest.raises(ValueError, match=r"GroupKFold5.*0\.\.4.*--bench-prepare"):
        _rebuild_prepared(prepared, folds=invalid_folds, manifest=manifest)


def test_save_rejects_forged_identity_before_creating_artifacts(tmp_path):
    prepared = _tiny_prepared()
    forged_prep_id = "0" * 16
    forged = PreparedDataset(
        prep_id=forged_prep_id,
        keys=prepared.keys,
        X=prepared.X,
        y_clf=prepared.y_clf,
        y_reg=prepared.y_reg,
        folds=prepared.folds,
        manifest=prepared.manifest,
        quality=prepared.quality,
    )
    output_dir = tmp_path / "prepared"

    with pytest.raises(ValueError, match="prep_id.*--bench-prepare"):
        forged.save(output_dir)

    assert not (output_dir / forged_prep_id).exists()


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


def test_groupkfold5_interpretation_describes_interleaved_factor_groups():
    from swmm_resilience.ml.bench.schemas import PROTOCOL_INTERPRETATION

    text = PROTOCOL_INTERPRETATION["GroupKFold5"]
    assert "intercalados" in text
    assert "ningún pliegue retiene dos factores contiguos" in text
    assert "bloques" not in text
