import hashlib
from contextlib import nullcontext
from pathlib import Path
import shutil
import sqlite3

import pandas as pd
import pytest

from swmm_resilience.database.training_queries import (
    IDENTITY_COLUMNS,
    TARGET_COLUMNS,
    load_training_frame,
)
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


def test_returns_canonical_27_column_frame(sql_training_db, csv_shaped_dataset):
    frame = load_training_frame(sql_training_db)

    assert frame.columns.tolist() == (
        list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS_V17) + list(TARGET_COLUMNS)
    )
    assert len(frame.columns) == 27
    assert len(frame) == len(csv_shaped_dataset)
    assert "coord_x" not in frame.columns
    assert "coord_y" not in frame.columns


def test_shared_columns_match_the_csv_shaped_source(
    sql_training_db, csv_shaped_dataset
):
    shared = [
        column
        for column in csv_shaped_dataset.columns
        if column not in ("coord_x", "coord_y")
    ]
    sort_keys = ["shape_id", "factor_mult", "node_id"]

    frame = load_training_frame(sql_training_db)

    actual = (
        frame[shared].sort_values(sort_keys).reset_index(drop=True)
    )
    expected = (
        csv_shaped_dataset[shared].sort_values(sort_keys).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_raises_for_a_database_with_no_complete_samples(tmp_path):
    db_path = tmp_path / "empty.sqlite3"
    with pytest.raises(ValueError, match="does not exist"):
        load_training_frame(db_path)
    assert not db_path.exists()


@pytest.mark.parametrize("reader", ["load_training_frame", "prepare_dataset"])
def test_reader_missing_migration_is_read_only(tmp_path, reader, monkeypatch):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    sql_dir = Path(__file__).parents[2] / "swmm_resilience" / "database" / "sql"
    migration_dir = tmp_path / "m"
    migration_dir.mkdir()
    for migration in sorted(sql_dir.glob("00[1-7]_*.sql")):
        shutil.copyfile(migration, migration_dir / migration.name)

    db_path = tmp_path / "schema_7.sqlite3"
    conn = connect_managed_database(db_path)
    apply_migrations(conn, migration_dir=migration_dir)
    conn.close()

    def schema_fingerprint():
        readonly = sqlite3.connect(
            f"file:{db_path.as_posix()}?mode=ro", uri=True
        )
        try:
            schema_rows = readonly.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "ORDER BY type, name, tbl_name"
            ).fetchall()
            return hashlib.sha256(repr(schema_rows).encode("utf-8")).hexdigest()
        finally:
            readonly.close()

    before_bytes = hashlib.sha256(db_path.read_bytes()).hexdigest()
    before_schema = schema_fingerprint()
    output_dir = tmp_path / "prepared"

    with pytest.raises(ValueError, match="missing migration.*008"):
        if reader == "load_training_frame":
            load_training_frame(db_path)
        else:
            from swmm_resilience.ml.bench.preprocess import prepare_dataset
            import swmm_resilience.ml.bench.preprocess as preprocess

            snapshot_dir = tmp_path / "snap"
            snapshot_dir.mkdir()
            monkeypatch.setattr(
                preprocess,
                "TemporaryDirectory",
                lambda prefix: nullcontext(snapshot_dir),
            )

            prepare_dataset(
                db_path,
                flood_threshold_m3=1.0,
                output_dir=output_dir,
            )

    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_bytes
    assert schema_fingerprint() == before_schema
    assert not output_dir.exists()


def test_repeated_calls_return_equal_frames(sql_training_db):
    first = load_training_frame(sql_training_db)
    second = load_training_frame(sql_training_db)

    pd.testing.assert_frame_equal(first, second)


def test_raises_when_two_networks_yield_duplicate_snapshot_keys(
    tmp_path, csv_shaped_dataset
):
    """Back-filling the same logical dataset against two different .inp files
    creates a second network (different network_sha256) whose scenarios/runs
    duplicate the (shape_id, factor_mult, node_id) keys of the first. The
    loader must raise rather than silently hand back a doubled frame."""
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.csv_backfill import backfill_networks_and_runs
    from swmm_resilience.database.migrations import apply_migrations

    db_path = tmp_path / "training_v17.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        inp_a = tmp_path / "network_a.inp"
        inp_a.write_text("[TITLE]\nnetwork a\n", encoding="utf-8")
        backfill_networks_and_runs(conn, csv_shaped_dataset, inp_a, "Network A")

        inp_b = tmp_path / "network_b.inp"
        inp_b.write_text("[TITLE]\nnetwork b (different bytes)\n", encoding="utf-8")
        backfill_networks_and_runs(conn, csv_shaped_dataset, inp_b, "Network B")
    finally:
        conn.close()

    with pytest.raises(ValueError) as excinfo:
        load_training_frame(db_path)

    message = str(excinfo.value)
    assert "multiple networks" in message
    assert "network_ids" in message


def test_explicit_network_selection_keeps_same_named_snapshots_distinct(
    tmp_path, csv_shaped_dataset
):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.csv_backfill import backfill_networks_and_runs
    from swmm_resilience.database.migrations import apply_migrations

    db_path = tmp_path / "two_networks.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        network_ids = []
        for name in ("A", "B"):
            inp_path = tmp_path / f"network_{name}.inp"
            inp_path.write_text(f"[TITLE]\\nnetwork {name}\\n", encoding="utf-8")
            info = backfill_networks_and_runs(conn, csv_shaped_dataset, inp_path, name)
            network_ids.append(info["network_id"])
    finally:
        conn.close()

    frame = load_training_frame(db_path, network_ids=network_ids)

    assert len(frame) == 2 * len(csv_shaped_dataset)
    assert set(frame["network_id"]) == set(network_ids)


def test_duplicate_snapshot_diagnostic_includes_network_id():
    from swmm_resilience.database.training_queries import _check_single_snapshot

    frame = pd.DataFrame(
        [
            {"network_id": 3, "scenario_id": 11, "node_id": "N0"},
            {"network_id": 3, "scenario_id": 11, "node_id": "N0"},
            {"network_id": 8, "scenario_id": 29, "node_id": "N0"},
        ]
    )

    with pytest.raises(ValueError) as excinfo:
        _check_single_snapshot(frame, allow_multiple_networks=True)

    assert "network_id=3" in str(excinfo.value)
    assert "scenario_id=11" in str(excinfo.value)
