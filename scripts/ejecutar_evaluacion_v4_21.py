"""Evaluación fuera de fold de XGBoost y de las referencias sin aprendizaje
sobre el contrato ``tabular_v4_21`` con los tres protocolos de ``config.yaml``
(hallazgos M-01, M-02, M-03 de la auditoría).

Uso (desde la raíz del repositorio):
    venv/Scripts/python.exe scripts/ejecutar_evaluacion_v4_21.py

Qué hace, en orden:
1. Prepara el dataset ``tabular_v4_21`` (lectura de SQLite vía snapshot de
   solo lectura) con ``bench.protocols`` en ``outputs/bench/prepared_v4_21``.
2. Toma los descriptores de forma (``duracion_horas``, ``tiempo_al_pico_h``)
   del PreparedDataset ``tabular_v3_17`` existente, que sí los trae, para la
   referencia de forma más cercana (el contrato v4_21 no los incluye).
3. Evalúa XGBoost con los hiperparámetros de ``config.yaml``
   (``bench.families.xgboost``, sin tuning de Optuna) y las dos referencias.
4. Escribe los OOF en ``outputs/bench/oof/<prep_id>/``, las métricas y la
   procedencia en ``docs/auditoria/correcciones/datos/runs/`` y las cuatro
   tablas del informe en ``docs/auditoria/correcciones/datos/``.

Nunca escribe en ``outputs/models``, ``outputs/bench/reports``,
``outputs/bench/prepared``, ``outputs/training_v17*.sqlite3`` ni ``data/``:
``ensure_allowed_output`` lo verifica antes de cada escritura.
"""

from __future__ import annotations

import fnmatch
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as _package_version
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.informe_evaluacion import build_tables, write_tables  # noqa: E402
from swmm_resilience.ml.bench.reports import _json_safe  # noqa: E402

SHAPE_DESCRIPTOR_COLUMNS = ("duracion_horas", "tiempo_al_pico_h")
V17_PREP_ID = "2db0578f875526cb"

PREPARED_DIR = Path("outputs/bench/prepared_v4_21")
OOF_ROOT = Path("outputs/bench/oof")
DATA_DIR = Path("docs/auditoria/correcciones/datos")
RUNS_DIR = DATA_DIR / "runs"
SHAPE_SOURCE_DIR = Path("outputs/bench/prepared") / V17_PREP_ID

_PROTECTED_DIRS = (
    Path("outputs/models"),
    Path("outputs/bench/reports"),
    Path("outputs/bench/prepared"),
    Path("data"),
)
_PROTECTED_FILE_PATTERN = "outputs/training_v17*.sqlite3*"

XGBOOST_PARAMS_NOTE = (
    "Los hiperparámetros de XGBoost son los valores por defecto de config.yaml "
    "(bench.families.xgboost), no los del modelo ajustado con Optuna: esta corrida "
    "mide el protocolo y las métricas, no el modelo promovido."
)


def shape_descriptors_from_frames(keys: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Tabla (índice ``shape_id``) con los descriptores de forma, una fila por forma.

    Falla si algún descriptor no es constante dentro de una forma o tiene NaN.
    """
    missing = [column for column in SHAPE_DESCRIPTOR_COLUMNS if column not in X.columns]
    if missing:
        raise ValueError(f"Faltan los descriptores de forma {missing} en las features.")
    frame = pd.DataFrame(
        {"shape_id": keys["shape_id"].astype(str).to_numpy()}
        | {column: X[column].to_numpy(dtype=float) for column in SHAPE_DESCRIPTOR_COLUMNS}
    )
    grouped = frame.groupby("shape_id")[list(SHAPE_DESCRIPTOR_COLUMNS)]
    varying = grouped.nunique(dropna=False) > 1
    if varying.to_numpy().any():
        shapes = sorted(varying.index[varying.any(axis=1)])
        raise ValueError(
            f"Los descriptores de forma no son constantes dentro de las formas {shapes}."
        )
    table = grouped.first()
    if table.isna().to_numpy().any():
        raise ValueError("Los descriptores de forma tienen NaN.")
    return table


def check_same_shapes(descriptors: pd.DataFrame, keys: pd.DataFrame) -> None:
    """Falla si las formas de ``descriptors`` y de ``keys`` no son las mismas."""
    have = set(descriptors.index.astype(str))
    want = set(keys["shape_id"].astype(str))
    if have != want:
        raise ValueError(
            "Los descriptores no cubren exactamente las mismas formas que el dataset: "
            f"faltan {sorted(want - have)}, sobran {sorted(have - want)}."
        )


def ensure_allowed_output(path: Path | str, repo_root: Path | str = REPO_ROOT) -> None:
    """Falla si ``path`` cae en una ruta protegida (modelos, reportes, datos fuente)."""
    root = Path(repo_root).resolve()
    target = Path(path)
    target = (target if target.is_absolute() else root / target).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError:
        return
    for protected in _PROTECTED_DIRS:
        if relative == protected or relative.is_relative_to(protected):
            raise ValueError(f"Ruta protegida, no se escribe aquí: {relative.as_posix()}")
    if fnmatch.fnmatch(relative.as_posix(), _PROTECTED_FILE_PATTERN):
        raise ValueError(f"Ruta protegida, no se escribe aquí: {relative.as_posix()}")


def _versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "pyarrow"):
        try:
            versions[name] = _package_version(name)
        except Exception:
            versions[name] = "unknown"
    return versions


def build_provenance(
    prepared,
    *,
    contract_id: str,
    flood_threshold_m3: float,
    max_continuity_error_pct: float | None,
    xgboost_params: dict,
    baselines: list[str],
    shape_descriptors_source: str,
    stage_seconds: dict,
    git_commit: str,
) -> dict:
    manifest = prepared.manifest
    return {
        "prep_id": prepared.prep_id,
        "feature_contract_id": contract_id,
        "n_features": int(prepared.X.shape[1]),
        "feature_columns": list(prepared.X.columns),
        "n_rows": int(len(prepared.keys)),
        "n_shapes": int(prepared.keys["shape_id"].nunique()),
        "n_factors": int(prepared.keys["factor_mult"].nunique()),
        "protocols": sorted(prepared.folds["protocol"].unique()),
        "flood_threshold_m3": float(flood_threshold_m3),
        "max_continuity_error_pct": max_continuity_error_pct,
        "source_snapshot_sha256": manifest.get("source_snapshot_sha256"),
        "training_frame_sha256": manifest.get("training_frame_sha256"),
        "xgboost_params": xgboost_params,
        "xgboost_params_note": XGBOOST_PARAMS_NOTE,
        "baselines": list(baselines),
        "shape_descriptors_source": shape_descriptors_source,
        "stage_seconds": dict(stage_seconds),
        "git_commit": git_commit,
        "versions": _versions(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/ejecutar_evaluacion_v4_21.py",
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _stage(name: str, stage_seconds: dict, func, *args, **kwargs):
    start = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] {name}...", flush=True)
    result = func(*args, **kwargs)
    stage_seconds[name] = round(time.perf_counter() - start, 2)
    print(f"[{datetime.now():%H:%M:%S}] {name}: {stage_seconds[name]} s", flush=True)
    return result


def main() -> int:
    from swmm_resilience.config import load_config
    from swmm_resilience.ml.bench.baselines import (
        BASELINES,
        FORMA_MAS_CERCANA,
        evaluate_baseline,
    )
    from swmm_resilience.ml.bench.evaluate import evaluate_candidate
    from swmm_resilience.ml.bench.preprocess import prepare_dataset
    from swmm_resilience.ml.bench.reports import write_metrics_files, write_oof
    from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21

    config = load_config(str(REPO_ROOT / "config.yaml"))
    xgboost_spec = config.bench.families["xgboost"]
    xgboost_params = {
        "classifier": dict(xgboost_spec.classifier),
        "regressor": dict(xgboost_spec.regressor),
    }
    threshold = config.dataset.flood_threshold_m3
    continuity = config.validation.max_continuity_error_pct
    for directory in (PREPARED_DIR, OOF_ROOT, DATA_DIR, RUNS_DIR):
        ensure_allowed_output(directory)
    git_commit = _git_commit()
    stage_seconds: dict[str, float] = {}

    prepared = _stage(
        "prepare_dataset",
        stage_seconds,
        prepare_dataset,
        config.dataset.db_path,
        protocols=tuple(config.bench.protocols),
        flood_threshold_m3=threshold,
        output_dir=REPO_ROOT / PREPARED_DIR,
        max_continuity_error_pct=continuity,
        feature_columns=FEATURE_COLUMNS_V21,
        feature_contract=TABULAR_V4_21,
        view_name="training_samples_v4_21",
        node_features_table="node_features_v4_21",
    )
    print(f"  prep_id={prepared.prep_id} filas={len(prepared.keys)} "
          f"features={prepared.X.shape[1]} protocolos={sorted(prepared.folds['protocol'].unique())}")

    source = REPO_ROOT / SHAPE_SOURCE_DIR
    descriptors = shape_descriptors_from_frames(
        pd.read_parquet(source / "keys.parquet"), pd.read_parquet(source / "features.parquet")
    )
    check_same_shapes(descriptors, prepared.keys)
    print(f"  descriptores de forma: {len(descriptors)} formas desde {SHAPE_SOURCE_DIR.as_posix()}")

    oof_by_name, metrics_by_name = {}, {}
    oof_by_name["xgboost"], metrics_by_name["xgboost"] = _stage(
        "xgboost",
        stage_seconds,
        evaluate_candidate,
        prepared,
        "xgboost",
        xgboost_params["classifier"],
        xgboost_params["regressor"],
    )
    for baseline in BASELINES:
        oof_by_name[baseline], metrics_by_name[baseline] = _stage(
            baseline,
            stage_seconds,
            evaluate_baseline,
            prepared,
            baseline,
            threshold,
            shape_descriptors=descriptors if baseline == FORMA_MAS_CERCANA else None,
        )

    for name, oof in oof_by_name.items():
        write_oof(oof, REPO_ROOT / OOF_ROOT, prepared.prep_id, name)
    write_metrics_files(metrics_by_name, REPO_ROOT / RUNS_DIR)

    tables = _stage("build_tables", stage_seconds, build_tables, prepared, oof_by_name, threshold)
    confusion = tables["matriz_confusion"]
    totals = confusion[["tn", "fp", "fn", "tp"]].sum(axis=1)
    if not (totals == len(prepared.keys)).all():
        raise AssertionError(f"La matriz de confusión no suma n_rows:\n{confusion}")
    write_tables(tables, REPO_ROOT / DATA_DIR)

    record = build_provenance(
        prepared,
        contract_id=TABULAR_V4_21.contract_id,
        flood_threshold_m3=threshold,
        max_continuity_error_pct=continuity,
        xgboost_params=xgboost_params,
        baselines=list(BASELINES),
        shape_descriptors_source=SHAPE_SOURCE_DIR.as_posix(),
        stage_seconds=stage_seconds,
        git_commit=git_commit,
    )
    record["oof_dir"] = (OOF_ROOT / prepared.prep_id).as_posix()
    record["prepared_dir"] = (PREPARED_DIR / prepared.prep_id).as_posix()
    (REPO_ROOT / RUNS_DIR / "provenance.json").write_text(
        json.dumps(_json_safe(record), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tables["comparacion_protocolos"].drop(columns=["que_mide"]).to_string(index=False))
        print(confusion.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
