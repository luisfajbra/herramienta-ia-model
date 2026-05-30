"""
High-level database query helpers — separated from repository.py so
SQL logic is testable in isolation.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..utils import new_id


def register_temporal_artifact(
    conn: sqlite3.Connection,
    run_id: str,
    network_hash: str,
    parquet_path: Path,
    node_count: int,
    step_count: int,
) -> str:
    """Insert a row into temporal_artifacts. Returns the artifact_id."""
    artifact_id = new_id()
    created_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO temporal_artifacts
            (artifact_id, run_id, network_hash, parquet_path,
             node_count, step_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact_id,
            run_id,
            network_hash,
            str(parquet_path),
            node_count,
            step_count,
            created_at,
        ),
    )
    conn.commit()
    return artifact_id
