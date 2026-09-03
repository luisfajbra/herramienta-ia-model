from pathlib import Path

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
    with pytest.raises(ValueError, match="No COMPLETE v17 training samples found"):
        load_training_frame(tmp_path / "empty.sqlite3")


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
    assert "more than one" in message or "multiple" in message
    assert "single snapshot" in message
    assert "base" in message
    assert "1.0" in message or "1" in message
    assert "N0" in message
