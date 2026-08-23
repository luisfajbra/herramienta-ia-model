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

# These paths are deliberately OUTSIDE the pre-existing broad `outputs/*` and
# `*.db` rules in .gitignore. They exist only to prove that the new
# sqlite-sidecar-specific patterns (*.sqlite3, *.sqlite3-wal, *.sqlite3-shm,
# *.sqlite3-journal, *.sqlite, *.sqlite-journal, *.db-wal, *.db-shm,
# *.workflow.lock) are themselves load-bearing, independent of any other
# rule in the file.
SIDECAR_NAMES_OUTSIDE_BROAD_RULES = [
    "data/training/dataset.sqlite3",
    "data/training/dataset.sqlite3-wal",
    "swmm_resilience/dataset.sqlite3-shm",
    "dataset.sqlite3-journal",
    "dataset.sqlite",
    "dataset.sqlite-journal",
    "dataset.db-wal",
    "dataset.db-shm",
    "dataset.workflow.lock",
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


@pytest.mark.parametrize("relative_path", SIDECAR_NAMES_OUTSIDE_BROAD_RULES)
def test_sqlite_sidecar_paths_outside_broad_rules_are_git_ignored(relative_path, tmp_path):
    """These paths are not covered by outputs/* or *.db, so a pass here
    proves the new sqlite-specific .gitignore patterns actually work."""
    target = REPO_ROOT / relative_path
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(target)],
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"{relative_path} is not git-ignored"
