# swmm_resilience/database/workflow_lock.py
from __future__ import annotations

import os
from pathlib import Path


class WorkflowLockError(RuntimeError):
    pass


class WorkflowLock:
    """Advisory, cross-process exclusive lock over one database file.

    Used to serialize training/migration/recovery operations against a
    single SQLite database. Not a substitute for SQLite's own locking —
    this guards multi-statement Python-level workflows.
    """

    def __init__(self, database_path: str | Path):
        self._lock_path = Path(f"{database_path}.workflow.lock")
        self._fd: int | None = None

    def acquire(self) -> "WorkflowLock":
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR)
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise WorkflowLockError(
                        f"Workflow lock already held: {self._lock_path}"
                    ) from exc
            else:
                import fcntl

                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise WorkflowLockError(
                        f"Workflow lock already held: {self._lock_path}"
                    ) from exc
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "WorkflowLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
