"""Interrupted-run recovery operations.

These functions provide an explicit, auditable way to reconcile database
state after a training run's process was killed or crashed mid-flight,
leaving rows stuck in ``RUNNING``/``PENDING``. They perform only the status
transitions already permitted by the ``training_runs_valid_status_transition``
and ``model_evaluations_valid_status_transition`` triggers (Task 5) — no rows
are ever deleted.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

RECOVERY_REASON = "interrupted_run_recovery"
RECOVERY_MESSAGE = "Marked failed by interrupted-run recovery"


def recover_abandoned_training_run(conn: sqlite3.Connection, training_run_id: int) -> None:
    """Fail an abandoned RUNNING training run and its unresolved evaluations.

    The caller is responsible for holding the process-level ``WorkflowLock``
    before invoking this function; it does not acquire one itself.

    Raises:
        ValueError: if the training run does not exist, or its status is not
            ``RUNNING`` (i.e. it was never abandoned mid-run, or has already
            been resolved/promoted).
    """
    row = conn.execute(
        "SELECT status FROM training_runs WHERE training_run_id = ?",
        (training_run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Training run {training_run_id} does not exist")
    if row[0] != "RUNNING":
        raise ValueError(
            f"Training run {training_run_id} is {row[0]!r}; only a RUNNING "
            "run can be recovered"
        )

    completed_at_utc = datetime.now(timezone.utc).isoformat()

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """
            UPDATE model_evaluations
            SET status = 'FAILED', failure_type = ?, failure_message = ?
            WHERE training_run_id = ? AND status IN ('PENDING', 'RUNNING')
            """,
            (RECOVERY_REASON, RECOVERY_MESSAGE, training_run_id),
        )
        conn.execute(
            """
            UPDATE training_runs
            SET status = 'FAILED', failure_type = ?, failure_message = ?,
                completed_at_utc = ?
            WHERE training_run_id = ?
            """,
            (RECOVERY_REASON, RECOVERY_MESSAGE, completed_at_utc, training_run_id),
        )
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
