import re
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import apply_migrations


@pytest.fixture
def migrated_connection(tmp_path):
    conn = None
    try:
        conn = connect_database(tmp_path / "query-plans.sqlite3")
        apply_migrations(conn)
        yield conn
    finally:
        if conn is not None:
            conn.close()


def _query_plan_details(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[object, ...],
) -> tuple[str, ...]:
    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    return tuple(str(row[3]) for row in rows)


def _assert_named_index(details: tuple[str, ...], index_name: str) -> None:
    pattern = re.compile(rf"\bINDEX\s+{re.escape(index_name)}\b", re.IGNORECASE)
    assert any(pattern.search(detail) for detail in details), details


@pytest.mark.parametrize(
    ("sql", "params", "expected_index"),
    [
        (
            "SELECT * FROM node_timeseries WHERE run_id=? ORDER BY time_sec",
            (1,),
            "idx_timeseries_run_time",
        ),
        (
            """
            SELECT * FROM node_timeseries
            WHERE run_id=? AND node_pk=?
            ORDER BY step_index
            """,
            (1, 10),
            "idx_timeseries_run_node_step",
        ),
        (
            """
            SELECT * FROM scenarios
            WHERE network_id=? AND scenario_kind=?
            """,
            (1, "factor"),
            "idx_scenarios_network_kind",
        ),
        (
            """
            SELECT * FROM model_metrics
            WHERE owner_kind=? AND owner_id=? AND metric_name=?
            """,
            ("evaluation", 1, "roc_auc"),
            "idx_metrics_owner",
        ),
        (
            """
            SELECT model_id, selected_value FROM trained_models
            WHERE target=?
              AND feature_contract_id=?
              AND selected_metric=?
            """,
            ("inunda", "tabular_v3_17", "roc_auc"),
            "idx_models_target_contract_metric",
        ),
    ],
)
def test_critical_queries_use_named_indexes(
    migrated_connection,
    sql,
    params,
    expected_index,
):
    details = _query_plan_details(migrated_connection, sql, params)

    _assert_named_index(details, expected_index)


def _relation_operations(
    details: tuple[str, ...],
    operation: str,
    *relation_names: str,
) -> tuple[str, ...]:
    names = "|".join(re.escape(name) for name in relation_names)
    pattern = re.compile(rf"\b{operation}\s+(?:{names})\b", re.IGNORECASE)
    return tuple(detail for detail in details if pattern.search(detail))


def _assert_search_uses_keys(
    searches: tuple[str, ...],
    *key_names: str,
) -> None:
    assert searches
    for key_name in key_names:
        key_pattern = re.compile(
            rf"\b{re.escape(key_name)}\s*=\s*\?",
            re.IGNORECASE,
        )
        assert any(key_pattern.search(detail) for detail in searches), searches


def test_training_view_uses_keyed_searches_for_high_volume_children(
    migrated_connection,
):
    details = _query_plan_details(
        migrated_connection,
        "SELECT * FROM training_samples_v17 WHERE run_id IN (?, ?)",
        (1, 2),
    )

    run_searches = _relation_operations(details, "SEARCH", "r", "runs")
    feature_searches = _relation_operations(
        details,
        "SEARCH",
        "f",
        "node_features",
    )
    result_searches = _relation_operations(
        details,
        "SEARCH",
        "o",
        "node_results",
    )

    assert run_searches, details
    assert any(
        re.search(r"\b(?:rowid|run_id)\s*=\s*\?", detail, re.IGNORECASE)
        for detail in run_searches
    ), run_searches
    _assert_search_uses_keys(feature_searches, "run_id")
    _assert_search_uses_keys(result_searches, "run_id", "node_pk")
    assert not _relation_operations(
        details,
        "SCAN",
        "f",
        "node_features",
    ), details
    assert not _relation_operations(
        details,
        "SCAN",
        "o",
        "node_results",
    ), details
