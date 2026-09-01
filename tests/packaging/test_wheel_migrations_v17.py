# tests/packaging/test_wheel_migrations_v17.py
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_wheel_exposes_all_five_migrations_and_validator(tmp_path):
    dist_dir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    wheel = next(dist_dir.glob("*.whl"))

    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", str(wheel)], check=True)

    script = (
        "from importlib.resources import files\n"
        "root = files('swmm_resilience.database').joinpath('sql')\n"
        "names = sorted(p.name for p in root.iterdir() if p.name.endswith('.sql'))\n"
        "assert names == ["
        "'001_v17_initial.sql','002_model_integrity.sql',"
        "'003_model_integrity_guards.sql','004_training_run_identity.sql',"
        "'005_provenance_integrity.sql'], names\n"
        "from swmm_resilience.database import migration_005_validator\n"
        "assert hasattr(migration_005_validator, 'validate_before_005')\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [str(python), "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
