$ErrorActionPreference = "Stop"

$repoDir = Split-Path -Parent $PSScriptRoot
Set-Location $repoDir

$pythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

& $pythonBin -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Se requiere Python 3.13 (o define PYTHON_BIN)."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $pythonBin -m venv .venv
}

& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r requirements.txt
& .venv\Scripts\python.exe -m pytest tests --ignore=tests/desktop
& .venv\Scripts\python.exe main.py

Write-Host "Reproducción completada. Revisa data/training/dataset_final.csv y outputs/."
