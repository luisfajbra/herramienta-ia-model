#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-python3.13}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Error: se requiere Python 3.13 (o define PYTHON_BIN)." >&2
  exit 1
fi

python_version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_version" != "3.13" ]]; then
  echo "Error: se requiere Python 3.13; se encontro $python_version." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests --ignore=tests/desktop
.venv/bin/python main.py

echo "Reproduccion completada. Revisa data/training/dataset_final.csv y outputs/."
