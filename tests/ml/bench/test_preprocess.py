import json
import shutil
import sqlite3

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


def _csv_shaped_dataset_v17() -> pd.DataFrame:
    """Same scenario grid as conftest's ``csv_shaped_dataset_six_factors``
    fixture (6 distinct factor_mult values -- GroupKFold5 needs at least 5
    groups), kept local so ``_build_sql_db_v4_21`` can backfill the matching
    v17 rows (node_features/node_results) that ``backfill_tributary_features``
    then attaches tributary features to via the same run_id/node_pk keys."""
    rows = []
    for shape_id in ("base", "storm_a"):
        for factor in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            for node_idx in range(4):
                flooded = 1 if (node_idx < 2 and factor >= 2.0) else 0
                rows.append(
                    {
                        "node_id": f"N{node_idx}",
                        "elev_fondo": 100.0 + node_idx,
                        "prof_max": 1.5,
                        "n_tuberias_in": 1,
                        "n_tuberias_out": 1,
                        "diam_max_in": 0.2,
                        "diam_max_out": 0.2,
                        "pendiente_max_in": 0.01,
                        "pendiente_out": 0.01,
                        "base_inflow_lps": 5.0 + node_idx,
                        "dist_outfall_m": 100.0 + node_idx * 10,
                        "n_nodos_aguas_arriba": node_idx,
                        "q_pico_acum_base": 10.0,
                        "upstream_capacity_lps": 50.0,
                        "coord_x": float(node_idx),
                        "coord_y": float(node_idx * 2),
                        "factor_mult": factor,
                        "q_pico_nodo": 2.0 * factor,
                        "q_pico_acum_escalado": 10.0 * factor,
                        "duracion_horas": 3.0,
                        "tiempo_al_pico_h": 0.7,
                        "shape_id": shape_id,
                        "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                        "inunda": flooded,
                    }
                )
    return pd.DataFrame(rows)


def _csv_shaped_dataset_v4_21() -> pd.DataFrame:
    """A node_features_v4_21-shaped frame mirroring ``_csv_shaped_dataset_v17``'s
    scenario grid (same node/shape/factor combinations -- 6 distinct
    factor_mult values), so the same runs backfilled into
    node_features/node_results via ``backfill_networks_and_runs`` can also
    receive tributary features via ``backfill_tributary_features``."""
    rows = []
    for shape_id in ("base", "storm_a"):
        for factor in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            for node_idx in range(4):
                flooded = 1 if (node_idx < 2 and factor >= 2.0) else 0
                rows.append(
                    {
                        "node_id": f"N{node_idx}",
                        "elev_fondo": 100.0 + node_idx,
                        "prof_max": 1.5,
                        "n_tuberias_in": 1,
                        "n_tuberias_out": 1,
                        "diam_max_in": 0.2,
                        "diam_max_out": 0.2,
                        "pendiente_max_in": 0.01,
                        "pendiente_out": 0.01,
                        "base_inflow_lps": 5.0 + node_idx,
                        "dist_outfall_m": 100.0 + node_idx * 10,
                        "n_nodos_aguas_arriba": node_idx,
                        "q_pico_acum_base": 10.0,
                        "upstream_capacity_lps": 50.0,
                        "factor_mult": factor,
                        "q_pico_nodo": 2.0 * factor,
                        "q_pico_acum_escalado": 10.0 * factor,
                        "volumen_local_integrado_m3": 1.0 * factor,
                        "volumen_tributario_integrado_m3": 2.0 * factor,
                        "duracion_efectiva_tributario_h": 1.5,
                        "tiempo_primer_pico_tributario_h": 0.5,
                        "tiempo_sobre_capacidad_h": 0.2 * factor if flooded else None,
                        "integral_exceso_capacidad_m3": 0.3 * factor if flooded else None,
                        "shape_id": shape_id,
                        "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                        "inunda": flooded,
                    }
                )
    return pd.DataFrame(rows)


def _build_sql_db_v4_21(tmp_path):
    """Build a fresh migrated database holding both node_features (v17) and
    node_features_v4_21 (tributary contract) rows for the same runs, so
    training_samples_v4_21 resolves. Mirrors ``_build_sql_db`` but also
    calls ``backfill_tributary_features`` to populate node_features_v4_21."""
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.csv_backfill import (
        backfill_networks_and_runs,
        backfill_tributary_features,
    )
    from swmm_resilience.database.migrations import apply_migrations

    dataset_v17 = _csv_shaped_dataset_v17()
    dataset_v4_21 = _csv_shaped_dataset_v4_21()

    inp_path = tmp_path / "tabular_v4_21.inp"
    inp_path.write_text("[TITLE]\nfixture network\n", encoding="utf-8")
    db_path = tmp_path / "tabular_v4_21.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        info = backfill_networks_and_runs(conn, dataset_v17, inp_path, "Fixture Network")
        backfill_tributary_features(
            conn,
            dataset_v4_21,
            info["run_id_by_key"],
            info["node_pk_by_id"],
            info["network_id"],
        )
    finally:
        conn.close()
    return db_path


def test_prepare_dataset_reads_tabular_v4_21_when_requested(tmp_path):
    from swmm_resilience.ml.bench.preprocess import prepare_dataset
    from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21

    db_path = _build_sql_db_v4_21(tmp_path)
    prepared = prepare_dataset(
        db_path,
        protocols=("GroupKFold5",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
        feature_columns=FEATURE_COLUMNS_V21,
        feature_contract=TABULAR_V4_21,
        view_name="training_samples_v4_21",
        node_features_table="node_features_v4_21",
    )
    assert list(prepared.X.columns) == list(FEATURE_COLUMNS_V21)
    assert "duracion_horas" not in prepared.X.columns


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
    assert manifest["protocol_definitions"] == {
        "LOSO": {"group_column": "factor_mult"}
    }
    assert manifest["source_capture_method"] == "sqlite_backup_readonly_v1"
    assert len(manifest["source_snapshot_sha256"]) == 64
    assert manifest["training_frame_fingerprint_version"] == "training_frame_canonical_v1"
    assert len(manifest["training_frame_sha256"]) == 64
    assert manifest["feature_columns"] == list(FEATURE_COLUMNS_V17)
    assert "python_version" in manifest
    assert "library_versions" in manifest
    assert "created_at_utc" in manifest


def test_each_protocol_uses_its_declared_group_column(sql_training_db, tmp_path):
    prepared = prepare_dataset(
        sql_training_db,
        protocols=("LOSO", "LeaveOneShapeOut"),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )

    assert prepared.manifest["protocol_definitions"] == {
        "LOSO": {"group_column": "factor_mult"},
        "LeaveOneShapeOut": {"group_column": "shape_id"},
    }
    shape_folds = prepared.folds[
        prepared.folds["protocol"] == "LeaveOneShapeOut"
    ]
    assert shape_folds["group_column"].unique().tolist() == ["shape_id"]
    assert "shape_id" not in prepared.X.columns


def test_resolve_prepared_rejects_manifest_without_protocol_semantics(prepared, tmp_path):
    manifest_path = tmp_path / "prepared" / prepared.prep_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("protocol_definitions", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"semántica.*protocol"):
        resolve_prepared(tmp_path / "prepared", prepared.prep_id)


def test_resolve_prepared_rejects_incorrect_protocol_semantics(prepared, tmp_path):
    manifest_path = tmp_path / "prepared" / prepared.prep_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_definitions"]["LOSO"]["group_column"] = "shape_id"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"semántica.*protocol"):
        resolve_prepared(tmp_path / "prepared", prepared.prep_id)


def test_metadata_only_source_change_changes_snapshot_but_not_frame_identity(
    sql_training_db, tmp_path
):
    modified_db = tmp_path / "same_rows_different_bytes.sqlite3"
    shutil.copy2(sql_training_db, modified_db)
    connection = sqlite3.connect(modified_db)
    try:
        current_user_version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
        new_user_version = 0 if current_user_version != 0 else 1
        connection.execute(f"PRAGMA user_version = {new_user_version}")
        connection.commit()
        busy, _, _ = connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()
        assert busy == 0
    finally:
        connection.close()

    assert preprocess_module._sha256_file(sql_training_db) != preprocess_module._sha256_file(
        modified_db
    ), "la mutación de la fixture debe cambiar los bytes antes de preparar el dataset"

    original = prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "original",
    )
    modified = prepare_dataset(
        modified_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "modified",
    )

    assert len(original.manifest["source_snapshot_sha256"]) == 64
    assert len(modified.manifest["source_snapshot_sha256"]) == 64
    int(original.manifest["source_snapshot_sha256"], 16)
    int(modified.manifest["source_snapshot_sha256"], 16)
    assert original.manifest["run_ids"] == modified.manifest["run_ids"]
    assert original.manifest["n_rows"] == modified.manifest["n_rows"]
    assert original.manifest["protocol_definitions"] == modified.manifest[
        "protocol_definitions"
    ]
    assert original.manifest["source_snapshot_sha256"] != modified.manifest[
        "source_snapshot_sha256"
    ]
    assert original.manifest["training_frame_sha256"] == modified.manifest[
        "training_frame_sha256"
    ]
    assert original.prep_id == modified.prep_id


def test_prepare_captures_readonly_snapshot_and_never_loads_source(
    sql_training_db, tmp_path, monkeypatch
):
    source_before = preprocess_module._sha256_file(sql_training_db)
    seen_paths = []
    real_loader = preprocess_module.load_training_frame

    def capture_loader(path, max_continuity_error_pct=None, **kwargs):
        seen_paths.append(path)
        assert str(path) != str(sql_training_db)
        assert preprocess_module.Path(path).exists()
        return real_loader(
            path, max_continuity_error_pct=max_continuity_error_pct, **kwargs
        )

    monkeypatch.setattr(preprocess_module, "load_training_frame", capture_loader)
    prepared = prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )

    assert len(seen_paths) == 1
    assert preprocess_module._sha256_file(sql_training_db) == source_before
    assert prepared.manifest["source_capture_method"] == "sqlite_backup_readonly_v1"


def test_relevant_training_value_change_changes_logical_identity(
    sql_training_db, tmp_path, monkeypatch
):
    real_loader = preprocess_module.load_training_frame
    baseline = prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "baseline",
    )
    changed = real_loader(sql_training_db).copy()
    changed.loc[0, FEATURE_COLUMNS_V17[0]] = (
        float(changed.loc[0, FEATURE_COLUMNS_V17[0]]) + 1.0
    )
    monkeypatch.setattr(
        preprocess_module,
        "load_training_frame",
        lambda _path, max_continuity_error_pct=None, **kwargs: changed,
    )
    modified = prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "modified",
    )

    assert baseline.manifest["source_snapshot_sha256"] == modified.manifest[
        "source_snapshot_sha256"
    ]
    assert baseline.manifest["training_frame_sha256"] != modified.manifest[
        "training_frame_sha256"
    ]
    assert baseline.prep_id != modified.prep_id


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

    monkeypatch.setattr(
        preprocess_module,
        "load_training_frame",
        lambda db_path, max_continuity_error_pct=None, **kwargs: shuffled,
    )

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


def _dataset_with_continuity(
    csv_shaped_dataset, *, high_shape_id, high_factor, high_value, low_value=0.5
):
    """csv_shaped_dataset + continuity_error_pct: un run alto, el resto bajos."""
    dataset = csv_shaped_dataset.copy()
    dataset["continuity_error_pct"] = low_value
    is_high = (dataset["shape_id"] == high_shape_id) & (
        dataset["factor_mult"] == high_factor
    )
    assert is_high.any(), "el setup debe marcar al menos un run como alto"
    dataset.loc[is_high, "continuity_error_pct"] = high_value
    return dataset


def _shape_factor_pairs(keys):
    return {
        (shape_id, float(factor))
        for shape_id, factor in zip(keys["shape_id"], keys["factor_mult"])
    }


def test_prepare_dataset_drops_runs_above_the_continuity_threshold(
    csv_shaped_dataset, tmp_path
):
    """Fase C2 / revision final (C1): el umbral no basta con existir en SQL --
    el banco real (prepare_dataset) debe recibirlo y perder esas filas. Sin esta
    asercion el filtro puede quedar nunca invocado en produccion y el reporte de
    excluidos miente."""
    dataset = _dataset_with_continuity(
        csv_shaped_dataset, high_shape_id="storm_a", high_factor=3.0,
        high_value=-7.31,
    )
    db_path = _build_sql_db(tmp_path, dataset, "continuity")

    filtered = prepare_dataset(
        db_path, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "filtered",
        max_continuity_error_pct=5.0,
    )
    unfiltered = prepare_dataset(
        db_path, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "unfiltered",
    )

    assert ("storm_a", 3.0) in _shape_factor_pairs(unfiltered.keys)
    assert ("storm_a", 3.0) not in _shape_factor_pairs(filtered.keys), (
        "el run con |continuity_error_pct| = 7.31 > 5.0 debe salir del banco"
    )
    assert ("base", 3.0) in _shape_factor_pairs(filtered.keys)
    n_nodes = dataset["node_id"].nunique()
    assert filtered.manifest["n_rows"] == unfiltered.manifest["n_rows"] - n_nodes
    assert filtered.manifest["max_continuity_error_pct"] == 5.0
    assert filtered.prep_id != unfiltered.prep_id


def test_prepare_dataset_keeps_runs_with_null_continuity(csv_shaped_dataset, tmp_path):
    """Ruling "NULL = no excluido": una corrida sin medicion de continuidad
    (p.ej. backfill viejo) no se descarta."""
    dataset = csv_shaped_dataset.copy()
    dataset["continuity_error_pct"] = float("nan")
    db_path = _build_sql_db(tmp_path, dataset, "null_continuity")

    prepared = prepare_dataset(
        db_path, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
        max_continuity_error_pct=5.0,
    )

    assert prepared.manifest["n_rows"] == len(dataset)


def test_prepare_dataset_without_threshold_keeps_high_continuity_runs(
    csv_shaped_dataset, tmp_path
):
    """El default None preserva el comportamiento historico (sin filtrar)."""
    dataset = _dataset_with_continuity(
        csv_shaped_dataset, high_shape_id="storm_a", high_factor=3.0,
        high_value=-7.31,
    )
    db_path = _build_sql_db(tmp_path, dataset, "unfiltered_default")

    prepared = prepare_dataset(
        db_path, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )

    assert prepared.manifest["n_rows"] == len(dataset)
    assert prepared.manifest["max_continuity_error_pct"] is None
