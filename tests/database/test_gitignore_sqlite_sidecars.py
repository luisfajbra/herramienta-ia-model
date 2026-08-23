# tests/database/test_gitignore_sqlite_sidecars.py
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]

SIDECAR_NAMES = [
    "outputs/models.sqlite3",
    "outputs/models.sqlite3-wal",
    "outputs/models.sqlite3-shm",
    "outputs/models.sqlite3-journal",
    "outputs/models.sqlite",
    "outputs/models.sqlite-journal",
    "outputs/models.db-wal",
    "outputs/models.db-shm",
    "outputs/models.workflow.lock",
]


@pytest.mark.parametrize("relative_path", SIDECAR_NAMES)
def test_sqlite_sidecar_paths_are_git_ignored(relative_path, tmp_path):
    target = REPO_ROOT / relative_path
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(target)],
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"{relative_path} is not git-ignored"
