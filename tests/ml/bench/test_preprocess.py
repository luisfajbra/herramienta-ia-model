import json
import shutil

import pandas as pd
import pytest

from swmm_resilience.ml.bench import preprocess as preprocess_module
from swmm_resilience.ml.bench.preprocess import (
    compute_prep_id,
    prepare_dataset,
    resolve_prepared,
)
from swmm_resilience.ml.bench.schemas import KEY_COLUMNS
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


def _build_sql_db(tmp_path, dataset: pd.DataFrame, name: str):
    """Build a fresh migrated v17 database from ``dataset`` (mirrors sql_training_db)."""
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.csv_backfill import backfill_networks_and_runs
    from swmm_resilience.database.migrations import apply_migrations

    inp_path = tmp_path / f"{name}.inp"
    inp_path.write_text("[TITLE]\nfixture network\n", encoding="utf-8")
    db_path = tmp_path / f"{name}.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        backfill_networks_and_runs(conn, dataset, inp_path, "Fixture Network")
    finally:
        conn.close()
    return db_path


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


def test_a_version_lookup_failure_does_not_abort_prepare_dataset(
    sql_training_db, tmp_path, monkeypatch
):
    """importlib.metadata.version() raises PackageNotFoundError if a
    distribution is installed under a different name than the import name.
    That is a provenance detail destined for manifest.json, not something
    that should kill the whole prepare stage -- a failed lookup must degrade
    to "unknown" for that entry instead of propagating."""

    def _boom(name):
        raise Exception(f"no distribution found for {name!r}")

    monkeypatch.setattr(preprocess_module, "_package_version", _boom)

    prepared = prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )

    assert prepared.manifest["library_versions"]["scikit-learn"] == "unknown"


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


def test_prep_id_changes_when_row_count_changes(csv_shaped_dataset, tmp_path):
    """Two databases with the same run_ids but a different row count must not collide.

    Without n_rows in the hashed descriptor, this would silently overwrite one
    dataset's artifacts with the other's under save()'s shared prep_id directory.
    """
    full_db = _build_sql_db(tmp_path, csv_shaped_dataset, "full")
    reduced_dataset = (
        csv_shaped_dataset[csv_shaped_dataset["node_id"] != "N3"]
        .reset_index(drop=True)
    )
    reduced_db = _build_sql_db(tmp_path, reduced_dataset, "reduced")

    full = prepare_dataset(
        full_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared_full",
    )
    reduced = prepare_dataset(
        reduced_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared_reduced",
    )

    assert full.manifest["run_ids"] == reduced.manifest["run_ids"], (
        "the test setup should keep run_ids identical so the difference is isolated to n_rows"
    )
    assert full.manifest["n_rows"] != reduced.manifest["n_rows"]
    assert full.prep_id != reduced.prep_id


def test_prepare_dataset_sorts_rows_even_when_source_frame_is_shuffled(
    sql_training_db, tmp_path, monkeypatch
):
    """Pins the canonical sort itself, not just its idempotence.

    load_training_samples already emits ORDER BY run_id, node_id, so a test that
    merely re-sorts prepared.keys and compares it to itself would stay green even
    if prepare_dataset's own sort_values(...).reset_index(drop=True) were deleted.
    This test forces the source frame out of order via a monkeypatched
    load_training_frame, so only prepare_dataset's own sort can save it.
    """
    from swmm_resilience.database.training_queries import (
        load_training_frame as real_load_training_frame,
    )

    real_frame = real_load_training_frame(sql_training_db)
    shuffled = real_frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert list(shuffled["node_id"]) != list(real_frame["node_id"]), (
        "sanity check: the shuffle must actually reorder rows"
    )

    monkeypatch.setattr(preprocess_module, "load_training_frame", lambda db_path: shuffled)

    prepared = preprocess_module.prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )

    expected_keys = (
        shuffled.sort_values(["run_id", "node_id"])
        .reset_index(drop=True)
        .loc[:, list(KEY_COLUMNS)]
    )
    pd.testing.assert_frame_equal(prepared.keys, expected_keys)


def test_prep_id_is_independent_of_db_path(sql_training_db, tmp_path):
    copy_path = tmp_path / "copy_training_v17.sqlite3"
    shutil.copy2(sql_training_db, copy_path)

    original = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared_original",
    )
    copied = prepare_dataset(
        copy_path, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared_copy",
    )
    assert original.prep_id == copied.prep_id
