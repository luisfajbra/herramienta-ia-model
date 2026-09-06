import json

import pandas as pd
import pytest

from swmm_resilience.ml.bench.preprocess import (
    compute_prep_id,
    prepare_dataset,
    resolve_prepared,
)
from swmm_resilience.ml.bench.schemas import KEY_COLUMNS
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


@pytest.fixture
def prepared(sql_training_db, tmp_path):
    return prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_features_follow_the_contract_whitelist_and_order(prepared):
    assert tuple(prepared.X.columns) == FEATURE_COLUMNS_V17
    assert tuple(prepared.keys.columns) == KEY_COLUMNS
    assert "coord_x" not in prepared.X.columns


def test_targets_come_from_sql_and_are_not_re_derived(sql_training_db, tmp_path):
    """inunda ya viene persistido; recalcularlo podria discrepar de la base."""
    prepared = prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=999999.0,   # umbral absurdo
        output_dir=tmp_path / "prepared",
    )
    assert prepared.y_clf.sum() > 0, "inunda debe venir de SQL, no derivarse del umbral"
    assert prepared.quality["threshold_disagreements"] > 0


def test_rows_are_sorted_canonically_by_run_id_then_node_id(prepared):
    ordered = prepared.keys.sort_values(["run_id", "node_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(prepared.keys, ordered)


def test_prep_id_is_stable_across_two_identical_runs(sql_training_db, tmp_path):
    first = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "a",
    )
    second = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "b",
    )
    assert first.prep_id == second.prep_id


def test_prep_id_changes_when_the_threshold_changes(sql_training_db, tmp_path):
    first = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "a",
    )
    second = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=2.5,
        output_dir=tmp_path / "b",
    )
    assert first.prep_id != second.prep_id


def test_prep_id_changes_when_the_protocol_set_changes():
    base = {
        "contract_id": "tabular_v3_17",
        "feature_contract_sha256": "a" * 64,
        "run_ids": [1, 2, 3],
        "feature_columns": list(FEATURE_COLUMNS_V17),
        "flood_threshold_m3": 1.0,
        "sort_key": ["run_id", "node_id"],
        "protocols": ["LOSO"],
    }
    other = dict(base, protocols=["LOSO", "GroupKFold5"])
    assert compute_prep_id(base) != compute_prep_id(other)


def test_prep_id_is_sixteen_hex_characters(prepared):
    assert len(prepared.prep_id) == 16
    int(prepared.prep_id, 16)


def test_manifest_records_full_provenance(prepared):
    manifest = prepared.manifest
    assert manifest["prep_id"] == prepared.prep_id
    assert manifest["contract_id"] == "tabular_v3_17"
    assert len(manifest["feature_contract_sha256"]) == 64
    assert manifest["protocols"] == ["LOSO"]
    assert manifest["feature_columns"] == list(FEATURE_COLUMNS_V17)
    assert "python_version" in manifest
    assert "library_versions" in manifest
    assert "created_at_utc" in manifest


def test_artifacts_are_written_to_disk(prepared, tmp_path):
    directory = tmp_path / "prepared" / prepared.prep_id
    for filename in (
        "keys.parquet", "features.parquet", "targets.parquet",
        "folds.parquet", "manifest.json", "quality_report.json",
    ):
        assert (directory / filename).exists(), f"falta {filename}"
    assert json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def test_folds_cover_every_sample_exactly_once_per_protocol(prepared):
    tested = prepared.folds.loc[prepared.folds["split"] == "test", "sample_idx"]
    assert sorted(tested) == list(range(len(prepared.X)))


def test_resolve_prepared_loads_the_only_dataset_when_no_id_given(prepared, tmp_path):
    loaded = resolve_prepared(tmp_path / "prepared")
    assert loaded.prep_id == prepared.prep_id


def test_resolve_prepared_errors_clearly_when_none_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="--bench-prepare"):
        resolve_prepared(tmp_path / "vacio")
