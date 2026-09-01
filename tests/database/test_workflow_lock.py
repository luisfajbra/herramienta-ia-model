# tests/database/test_workflow_lock.py
import multiprocessing
import time

import pytest

from swmm_resilience.database.workflow_lock import WorkflowLock, WorkflowLockError


def test_second_acquire_in_same_process_fails(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    with WorkflowLock(db_path):
        with pytest.raises(WorkflowLockError):
            with WorkflowLock(db_path):
                pass


def test_lock_is_released_on_exit(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    with WorkflowLock(db_path):
        pass
    with WorkflowLock(db_path):
        pass  # must not raise; prior lock was released


def _hold_lock_then_signal(db_path, ready_event, release_event):
    with WorkflowLock(db_path):
        ready_event.set()
        release_event.wait(timeout=5)


def test_second_process_cannot_acquire_held_lock(tmp_path):
    db_path = tmp_path / "db.sqlite3"
    ready = multiprocessing.Event()
    release = multiprocessing.Event()
    holder = multiprocessing.Process(
        target=_hold_lock_then_signal, args=(db_path, ready, release)
    )
    holder.start()
    try:
        assert ready.wait(timeout=5)
        with pytest.raises(WorkflowLockError):
            with WorkflowLock(db_path):
                pass
    finally:
        release.set()
        holder.join(timeout=5)
