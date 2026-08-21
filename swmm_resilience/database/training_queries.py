from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd

from ..ml.contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17


IDENTITY_COLUMNS = (
    "run_id",
    "network_id",
    "scenario_id",
    "scenario_key",
    "scenario_kind",
    "factor_mult",
    "shape_id",
    "node_id",
)
TARGET_COLUMNS = ("inunda", "vol_inundacion_m3")


def _run_id_selection(
    run_ids: list[int] | None,
) -> tuple[str, list[int]]:
    if run_ids is None:
        return "", []
    if not run_ids:
        raise ValueError("run_ids cannot be empty")
    placeholders = ",".join("?" for _ in run_ids)
    return f" AND run_id IN ({placeholders})", list(run_ids)


def _find_key_mismatches(
    conn: sqlite3.Connection,
    run_ids: list[int] | None,
) -> list[sqlite3.Row]:
    run_filter, params = _run_id_selection(run_ids)
    return conn.execute(
        f"""
        WITH selected_runs AS (
            SELECT run_id
            FROM runs
            WHERE status = 'COMPLETE'{run_filter}
        ),
        mismatches AS (
            SELECT
                'missing result' AS mismatch,
                feature.run_id,
                feature.node_pk
            FROM node_features AS feature
            JOIN selected_runs AS selected
                ON selected.run_id = feature.run_id
            LEFT JOIN node_results AS result
                ON result.run_id = feature.run_id
                AND result.node_pk = feature.node_pk
            WHERE result.run_id IS NULL

            UNION ALL

            SELECT
                'missing feature' AS mismatch,
                result.run_id,
                result.node_pk
            FROM node_results AS result
            JOIN selected_runs AS selected
                ON selected.run_id = result.run_id
            LEFT JOIN node_features AS feature
                ON feature.run_id = result.run_id
                AND feature.node_pk = result.node_pk
            WHERE feature.run_id IS NULL
        )
        SELECT mismatch, run_id, node_pk
        FROM mismatches
        ORDER BY run_id, node_pk, mismatch
        LIMIT 20
        """,
        params,
    ).fetchall()


def load_training_samples(
    conn: sqlite3.Connection,
    run_ids: list[int] | None = None,
) -> pd.DataFrame:
    run_filter, params = _run_id_selection(run_ids)
    mismatches = _find_key_mismatches(conn, run_ids)
    if mismatches:
        details = ", ".join(
            f"{row[0]} ({row[1]}, {row[2]})" for row in mismatches
        )
        raise ValueError(
            "feature/result row count or key mismatch for COMPLETE runs: "
            f"{details}"
        )
    columns = IDENTITY_COLUMNS + FEATURE_COLUMNS_V17 + TARGET_COLUMNS
    sql = (
        f"SELECT {', '.join(columns)} FROM training_samples_v17"
        f" WHERE 1 = 1{run_filter}"
        " ORDER BY run_id, node_id"
    )
    frame = pd.read_sql_query(sql, conn, params=params)
    if frame.empty:
        raise ValueError("No COMPLETE v17 training samples found")
    TABULAR_V3_17.validate_frame(frame.loc[:, FEATURE_COLUMNS_V17])
    return frame


def export_training_samples_csv(
    conn: sqlite3.Connection,
    output_path: str | Path,
    run_ids: list[int] | None = None,
) -> Path:
    frame = load_training_samples(conn, run_ids=run_ids)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output
